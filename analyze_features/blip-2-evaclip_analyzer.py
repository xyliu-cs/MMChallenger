import torch
from transformers import Blip2Processor, Blip2Model
from PIL import Image
import os, json
from tqdm import tqdm
from lemminflect import getInflection, getLemma
from eva_clip import build_eva_model_and_transforms
from clip import tokenize
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
import analyze_utils as utils



eva_clip_path = "eva_clip_psz14.pt" # 
model_name = "EVA_CLIP_g_14"
image_path = "CLIP.png"
caption = ["a diagram", "a dog", "a cat"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = build_eva_model_and_transforms(model_name, pretrained=eva_clip_path)
model = model.to(device)

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = tokenize(caption).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs) 




class blip2_feature_analyzer:
    def __init__(self, config: dict) -> None:
        # this clip model should be modified to use -2 hidden layer's output as visual feature to align with blip-2 setting
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.evaclip, self.evaclip_preprocessor = build_eva_model_and_transforms(config["eva_clip_name"], pretrained=config["eva_clip_path"]).to(self.device)
        self.evaclip_tokenizer = tokenize
        self.blip2_processor = Blip2Processor.from_pretrained(config["blip2_path"])
        with init_empty_weights():
            blip2_t5_xxl = Blip2Model.from_pretrained(config["blip2_path"], torch_dtype=torch.float16)
            # change maximum mem here
            dmap = infer_auto_device_map(blip2_t5_xxl, max_memory={0: "20GiB", 1: "20GiB"}, no_split_module_classes=["T5Block"])
            # print(dmap)
            dmap['language_model.lm_head'] = dmap['language_projection'] = dmap['language_model.decoder.embed_tokens']
        blip2 = load_checkpoint_and_dispatch(blip2_t5_xxl, config["blip2_path"], device_map=dmap)
        self.blip2 = blip2.eval()

        self.challset_folder = os.path.normpath(config['challset_folder'])
        self.index_file = config['index_file_name']
        self.prepared_inputs = config['prepared_inputs_name']
        self.dummy_cap = config['dummy_cap']

        self.vision_features = []
        self.qformer_features = []
        self.language_features = []
        
    def prepare_input_pairs(self) -> None:
        index_file_path = os.path.join(self.challset_folder, self.index_file)
        with open(index_file_path, 'r') as f:
            error_list = json.load(f)
        
        ret_list = []
        for error_dict in error_list:
            category = error_dict["category"]
            assert category in ['action', 'place'], f"Unsupported category {category}"
            model_ans_list = error_dict["sa_model_ans"] # we can change type here
            false_caps = []
            if category == 'action':
                true_cap = f"{error_dict['context']["subject"]} {error_dict['context']["place"]} {error_dict['target']["action"]}"
                for ans_list in model_ans_list:
                    phrase = utils.convert_to_vbg(ans_list[0]).lower()
                    false_caps.append(f"{error_dict['context']["subject"]} {error_dict['context']["place"]} {phrase}")
            elif category == 'place':
                true_cap = f"{error_dict['context']["subject"]} {error_dict['context']["action"]} {error_dict['target']["place"]}"
                for ans_list in model_ans_list:
                    phrase = ans_list[0].lower()
                    false_caps.append(f"{error_dict['context']["subject"]} {error_dict['context']["action"]} {phrase}")

            # print('false caps: ', false_caps)
            image_names = error_dict["image"]
            for i, name in enumerate(image_names):
                # image_obj = Image.open(os.path.join(self.challset_folder, name)).convert("RGB")
                false_cap = false_caps[i]
                ret_list.append((os.path.join(self.challset_folder, name), name, true_cap, false_cap, self.dummy_cap))

        prepared_file = os.path.join(self.challset_folder, self.prepared_inputs)
        utils.write_json(prepared_file, ret_list)    
    
    # TODO: Need to change the visual feature to use from last to -2
    def compare_clip_emb(self, image_text_pairs) -> list:
        scored_list = []
        
        for image_text_tup in tqdm(image_text_pairs[:], "Checking CLIP embeddings: "):
            text_caps = [image_text_tup[2], image_text_tup[3], image_text_tup[4]]      # [true, false, dummy]

            image = self.evaclip_preprocessor(Image.open(image_text_tup[0])).unsqueeze(0).to(self.device)
            text = self.evaclip_tokenizer(text_caps).to(self.device)

            with torch.no_grad():
                image_features = self.evaclip.encode_image(image)
                text_features = self.evaclip.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                consine_sim = (image_features @ text_features.T).squeeze()
            # Append pooled features
            self.vision_features.append((image_features, text_features))
            ret_tup = (image_text_tup[1], image_text_tup[2], image_text_tup[3], image_text_tup[4], consine_sim.tolist())
            scored_list.append(ret_tup)
        return scored_list


    def compare_qformer_emb(self, image_text_pairs, pooling_type='max') -> list:
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking qformer embeddings: "):
            image = utils.read_image(image_text_tup[0])
            text_caps = [image_text_tup[2], image_text_tup[3], image_text_tup[4]] # [true, false, dummy]

            image_inputs = self.blip2_processor.image_processor(images=image, return_tensors="pt")
            text_only_inputs = self.blip2_processor.tokenizer(text=text_caps, return_tensors="pt", padding=True)
            
            qformer_feature = self.blip2.get_qformer_features(**image_inputs)
            image_embeds = self.blip2.language_projection(qformer_feature)

            text_embeds = self.blip2.get_input_embeddings()(text_only_inputs.input_ids)

            # Apply pooling to the embeddings
            if pooling_type == 'max':
                pooled_image_embeds = torch.max(image_embeds, dim=1)[0]  
                pooled_text_embeds = torch.max(text_embeds, dim=1)[0]
            elif pooling_type == 'avg':
                pooled_image_embeds = torch.mean(image_embeds, dim=1)  
                pooled_text_embeds = torch.mean(text_embeds, dim=1) 
            else:
                raise ValueError("Invalid pooling type. Use 'max' or 'avg'.")

            # Normalize the pooled features
            pooled_image_embeds = pooled_image_embeds / pooled_image_embeds.norm(p=2, dim=-1, keepdim=True)  # Shape: (1, 5120)
            pooled_text_embeds = pooled_text_embeds / pooled_text_embeds.norm(p=2, dim=-1, keepdim=True)  # Shape: (2, 5120)
            
            # Append pooled features
            self.qformer_features.append((pooled_image_embeds, pooled_text_embeds))

            # Compute cosine similarity via matrix multiplication
            cosine_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)

            ret_tup = (image_text_tup[1], image_text_tup[2], image_text_tup[3], image_text_tup[4], cosine_sim.tolist())
            scored_list.append(ret_tup)
        return scored_list

    def compare_blip2_emb(self, image_text_pairs, pooling_type='max') -> list:
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking blip2 embeddings: "):
            image = utils.read_image(image_text_tup[0])
            text_caps = [image_text_tup[2], image_text_tup[3], image_text_tup[4]]    # [true, false, dummy]
            
            visual_only_inputs = self.blip2_processor(images=image, return_tensors="pt")
            text_only_inputs = self.blip2_processor.tokenizer(text=text_caps, return_tensors="pt", padding=True)
            
            image_encoder_lhs = self.blip2(**visual_only_inputs)
            text_encoder_lhs = self.blip2(**text_only_inputs)

            # Apply pooling to the embeddings
            if pooling_type == 'max':
                pooled_image_embeds = torch.max(image_encoder_lhs, dim=1)[0]  
                pooled_text_embeds = torch.max(text_encoder_lhs, dim=1)[0]
            elif pooling_type == 'avg':
                pooled_image_embeds = torch.mean(image_encoder_lhs, dim=1)  
                pooled_text_embeds = torch.mean(text_encoder_lhs, dim=1) 
            else:
                raise ValueError("Invalid pooling type. Use 'max' or 'avg'.")
            
            # Append pooled features
            self.language_features.append((pooled_image_embeds, pooled_text_embeds))
            # Compute cosine similarity via matrix multiplication
            cosine_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)

            ret_tup = (image_text_tup[1], image_text_tup[2], image_text_tup[3], image_text_tup[4], cosine_sim.tolist())
            scored_list.append(ret_tup)         
        return scored_list

    def __call__(self):
        self.prepare_input_pairs()
        input_list = utils.read_json(os.path.join(self.challset_folder, self.prepared_inputs))
        clip_emb_list = self.compare_clip_emb(input_list)
        pooling_type = 'avg'
        print("Stage 1 (CLIP) feature analysis completed.")
        mmp_emb_list_max = self.compare_qformer_emb(input_list, pooling_type=pooling_type)
        print(f"Stage 2 (Q-Former) feature analysis completed with {pooling_type} pooling.")
        llava_emb_list_max = self.compare_blip2_emb(input_list, pooling_type=pooling_type)
        print(f"Stage 3 (BLIP-2) feature analysis completed with {pooling_type} pooling.")

        # everything is fine, pack and go
        if len(clip_emb_list) == len(mmp_emb_list_max) == len(llava_emb_list_max):
            ret_list = []
            for i in range(len(clip_emb_list)):
                local_dict = {}
                clip_info_list = clip_emb_list[i]
                local_dict["image"] = clip_info_list[0]
                local_dict["true_cap"] = clip_info_list[1]
                local_dict["false_cap"] = clip_info_list[2]
                local_dict["dummy_cap"] = clip_info_list[3]

                local_dict["clip_sim_score"] = clip_info_list[4]
                local_dict[f"qformer_sim_score_{pooling_type}"] = mmp_emb_list_max[i][4]
                local_dict[f"blip2_sim_score_{pooling_type}"] = llava_emb_list_max[i][4]
                
                ret_list.append(local_dict)
            output = os.path.join(self.challset_folder, f'feature_sim_score_{pooling_type}_pooling.json')
            utils.write_json(output, ret_list)
        else:
            pass #TODO: finish the logic here
    

if __name__ == '__main__':
    config_path = 'blip-2-evaclip_analyzer_config.json'
    analyzer_config = utils.read_json(config_path)
    analyzer = blip2_feature_analyzer(analyzer_config)
    analyzer.prepare_input_pairs()
    # analyzer()