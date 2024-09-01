import torch
from transformers import Blip2Processor, Blip2Model
from PIL import Image
import os, json, sys, time
from tqdm import tqdm
from clip import tokenize
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
import analyze_utils as utils

sys.path.append('/120040051/Github_Repos/EVA/EVA-01/clip')
from eva_clip import build_eva_model_and_transforms


class blip2_feature_analyzer:
    def __init__(self, config: dict) -> None:

        self.evaclip_path = config["eva_clip_path"]
        self.evaclip_name = config["eva_clip_name"]
        self.blip2_path = config["blip2_path"]
        
        self.challset_folder = os.path.normpath(config['challset_folder'])
        self.index_file = config['index_file_name']
        self.prepared_inputs = config['prepared_inputs_name']
        
        self.dummy_cap = config['dummy_cap']
        self.action_exp = config['action_exp']
        self.place_exp = config['place_exp']

        # self.vision_features = []
        # self.qformer_features = []
        # self.language_features = []
        
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
                ret_list.append((os.path.join(self.challset_folder, name), name, category, true_cap, false_cap, self.dummy_cap))

        prepared_file = os.path.join(self.challset_folder, self.prepared_inputs)
        utils.write_json(prepared_file, ret_list)    


    # TODO: Need to change the visual feature to use from last to -2
    def compare_clip_emb(self, image_text_pairs) -> list:
        scored_list = []

        # this clip model should be modified to use -2 hidden layer's output as visual feature to align with blip-2 setting
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaclip, evaclip_preprocessor = build_eva_model_and_transforms(self.evaclip_name, pretrained=self.evaclip_path)
        evaclip = evaclip.to(device)
        evaclip_tokenizer = tokenize
        
        for image_text_tup in tqdm(image_text_pairs[:], "Checking CLIP embeddings: "):
            text_caps = image_text_tup[3:]      # [true, false, dummy]
            image_path = image_text_tup[0]
            
            image = evaclip_preprocessor(Image.open(image_path)).unsqueeze(0)
            text = evaclip_tokenizer(text_caps)
            image = image.to(device)
            text = text.to(device)
            
            with torch.no_grad():
                image_features = evaclip.encode_image(image)
                text_features = evaclip.encode_text(text)
            
            cosine_sim = utils.cosine_sim(image_features, text_features)
            ret_tup = image_text_tup[1:] + [cosine_sim.tolist()]
            scored_list.append(ret_tup)
        return scored_list


    def compare_adapter_emb(self, image_text_pairs, pooling_type='avg_eol', use_prompt=True, contextualize_text=True) -> list:
        blip2_processor = Blip2Processor.from_pretrained(self.blip2_path)
        with init_empty_weights():
            blip2_t5_xxl = Blip2Model.from_pretrained(self.blip2_path, torch_dtype=torch.float16)
        blip2 = load_checkpoint_and_dispatch(blip2_t5_xxl, self.blip2_path, device_map='auto', no_split_module_classes=["T5Block"])
        blip2 = blip2.eval()        
        
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking qformer embeddings: "):            
            image_path = image_text_tup[0]
            category = image_text_tup[2]
            
            image = utils.read_image(image_path)
            if use_prompt:  
                text_caps = [cap.capitalize() + '.' for cap in image_text_tup[3:]]      # [true, false, dummy]
                prompt_eol = 'This sentence : "{sent_exp}" means in one word:"{word_exp}". This sentence : "{actual_sent}" means in one word:"'
                if category == 'action':
                    text_query = [prompt_eol.format(sent_exp=self.action_exp[0], word_exp=self.action_exp[1], actual_sent=text_cap) for text_cap in text_caps]
                elif category == 'place':
                    text_query = [prompt_eol.format(sent_exp=self.place_exp[0], word_exp=self.place_exp[1], actual_sent=text_cap) for text_cap in text_caps]  
            else:
                text_query = image_text_tup[3:]
            
            blip2_processor.tokenizer.padding_side = 'left'
            
            image_inputs = blip2_processor.image_processor(images=image, return_tensors="pt")
            text_inputs = blip2_processor.tokenizer(text=text_query, return_tensors="pt", padding=True)
            print('text inputs:\n', image_inputs.input_ids)
            
            image_inputs = image_inputs.to(blip2.device)
            text_inputs = text_inputs.to(blip2.device)
            
            with torch.no_grad():
                if contextualize_text:
                    text_embeds = blip2(**text_inputs)    # encoder last hidden states, this requires modification of modeling_blip_2.py
                else:
                    text_embeds = blip2.get_input_embeddings()(text_inputs.input_ids)   
                qformer_outputs = blip2.get_qformer_features(**image_inputs, output_hidden_states=False)
                qformer_lhs = qformer_outputs[0]
                image_embeds = blip2.language_projection(qformer_lhs)

            # Apply pooling to the embeddings
            pooled_image_embeds, pooled_text_embeds = utils.custom_pooling(image_embeds, text_embeds, pooling_type)
            # Compute cosine similarity via matrix multiplication
            cosine_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)
            ret_tup = image_text_tup[1:] + [cosine_sim.tolist()]
            scored_list.append(ret_tup)
        return scored_list


    def compare_llm_emb(self, image_text_pairs, use_prompt=True, pooling_type='eol') -> list:
        blip2_processor = Blip2Processor.from_pretrained(self.blip2_path)

        blip2 = Blip2Model.from_pretrained(self.blip2_path, device_map='auto', torch_dtype=torch.float16)

        blip2 = blip2.eval()   
        blip2_processor.num_query_tokens = blip2.config.num_query_tokens
        blip2_processor.tokenizer.padding_side = 'left'
        
        
        # TODO: Unfinished here !!!!!
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking blip2 embeddings: "):
            image_path = image_text_tup[0]
            category = image_text_tup[2]
            
            image = utils.read_image(image_path)
            if use_prompt:  
                text_caps = [cap.capitalize() + '.' for cap in image_text_tup[3:]]      # [true, false, dummy]
                prompt_eol = 'This sentence : "{sent_exp}" means in one word:"{word_exp}". This sentence : "{actual_sent}" means in one word:"'
                #  copying the below comments from processing_blip_2.py from transformers 4.45.0 dev
                #  " if we know how many query tokens, expand text inside processor. We need this hacky manipulation
                #  because BLIP expects image tokens to be at the beginning even before BOS token "
                #  This means that blip model always puts the image token first, so it is a little awkward for us to use PromptEOL
                prompt_eol_img = 'This image means in one word:"'
                if category == 'action':
                    text_query = [prompt_eol.format(sent_exp=self.action_exp[0], word_exp=self.action_exp[1], actual_sent=text_cap) for text_cap in text_caps]
                elif category == 'place':
                    text_query = [prompt_eol.format(sent_exp=self.place_exp[0], word_exp=self.place_exp[1], actual_sent=text_cap) for text_cap in text_caps]
                
                text_inputs = blip2_processor.tokenizer(text=text_query, return_tensors="pt", padding=True)
                image_inputs = blip2_processor(text=prompt_eol_img, images=image, return_tensors="pt")
                print(image_inputs.input_ids)
                print("text input shape: ", text_inputs.input_ids[0].shape)
                print("image input shape: ", image_inputs.input_ids[0].shape) 
            else:
                text_query = image_text_tup[3:]  # [true, false, dummy]
                text_inputs = blip2_processor.tokenizer(text=text_query, return_tensors="pt", padding=True)
                image_inputs = blip2_processor(images=image, return_tensors="pt")
                print(image_inputs)  
            
            text_inputs.to(blip2.device)
            image_inputs.to(blip2.device)
            
            # the below code requires modification of the modeling_blip_2.py 
            image_encoder_lhs = blip2(**image_inputs)
            text_encoder_lhs = blip2(**text_inputs)

            print(image_encoder_lhs.shape)
            print(text_encoder_lhs.shape)
            # Apply pooling to the embeddings
            pooled_image_embeds, pooled_text_embeds = utils.custom_pooling(image_encoder_lhs, text_encoder_lhs, pooling_type)
            
            # Append pooled features
            # self.language_features.append((pooled_image_embeds, pooled_text_embeds))
            # Compute cosine similarity via matrix multiplication
            cosine_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)

            ret_tup = image_text_tup[1:] + [cosine_sim.tolist()]
            scored_list.append(ret_tup)         
        return scored_list

    def __call__(self):
        # self.prepare_input_pairs()
        input_list = utils.read_json(os.path.join(self.challset_folder, self.prepared_inputs))
        # clip_emb_list = self.compare_clip_emb(input_list)
        # print(clip_emb_list[:5])
        # print(f"Stage 1 (CLIP) feature analysis completed.")
        # torch.cuda.empty_cache()
        
        # adapter_pooling_type = 'avg_eol'  # or 'avg'
        # adapter_emb_list = self.compare_adapter_emb(input_list, pooling_type=adapter_pooling_type, use_prompt=True, contextualize_text=True)
        # print(f"Stage 2 (Adapter) feature analysis completed with {adapter_pooling_type} pooling.")
        # print(adapter_emb_list[:5])
        # torch.cuda.empty_cache()

        lm_pooling_type = 'eol'
        llm_emb_list = self.compare_llm_emb(input_list, use_prompt=True, pooling_type=lm_pooling_type)
        print(f"Stage 3 (LLM) feature analysis completed with {lm_pooling_type} pooling.")
        print(llm_emb_list[:20])
        torch.cuda.empty_cache()

        # # utils.write_json(output, qwen_emb_list)
        # # everything is fine, pack and go
        # if len(clip_emb_list) == len(adapter_emb_list) == len(llm_emb_list):
        #     ret_list = []
        #     for i in range(len(clip_emb_list)):
        #         clip_info_list = clip_emb_list[i]
        #         local_dict = {"image": clip_info_list[0], "category": clip_info_list[1],  "true_cap": clip_info_list[2],
        #                       "false_cap": clip_info_list[3], "dummy_cap": clip_info_list[4],
        #                       "clip_sim_score": clip_info_list[5], "adapter_sim_score": adapter_emb_list[i][5], "lm_sim_score": llm_emb_list[i][5]}
        #         ret_list.append(local_dict)
        #     output = os.path.join(self.challset_folder, f'blip2_image_text_sim_scores.json')
        #     utils.write_json(output, ret_list)
        # else:
        #     pass #TODO: finish the logic here
    

if __name__ == '__main__':
    config_path = 'blip-2-evaclip_analyzer_config.json'
    analyzer_config = utils.read_json(config_path)
    analyzer = blip2_feature_analyzer(analyzer_config)
    # analyzer.prepare_input_pairs()

    analyzer()