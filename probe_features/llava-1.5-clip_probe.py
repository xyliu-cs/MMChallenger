import torch
from transformers import AutoProcessor, CLIPModel, LlavaForConditionalGeneration
from PIL import Image
import os, json
from tqdm import tqdm


def read_json(json_path):
    print(f"Read json file from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(json_path: str, json_list: list):
    with open(json_path, 'w') as f:
        json.dump(json_list, f, indent=2)
    print(f"Write {len(json_list)} items to {json_path}")


class llava_feature_probe:
    def __init__(self, config: dict) -> None:
        # this clip model should be modified to use -2 hidden layer's output as visual feature to align with llava setting
        # this modified clip model should NOT utilize logit_scale (originally set to e^2.6592) in order to only output cosine similarities
        self.clip = CLIPModel.from_pretrained(config["clip_path"]).eval()
        self.clip_processor = AutoProcessor.from_pretrained(config["clip_path"])
        self.llava = LlavaForConditionalGeneration.from_pretrained(config["llava_path"], torch_dtype=torch.float16, device_map='auto').eval()
        self.llava_processor = AutoProcessor.from_pretrained(config["llava_path"])
        self.challset_folder = os.path.normpath(config['challset_folder'])
        self.index_file = config['index_file_name']

    def prepare_input_pairs(self) -> list:
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
                    phrase = ans_list[0].lower()
                    false_caps.append(f"{error_dict['context']["subject"]} {error_dict['context']["place"]} {phrase}")
            elif category == 'place':
                true_cap = f"{error_dict['context']["subject"]} {error_dict['context']["action"]} {error_dict['target']["place"]}"
                for ans_list in model_ans_list:
                    phrase = ans_list[0].lower()
                    false_caps.append(f"{error_dict['context']["subject"]} {error_dict['context']["action"]} {phrase}")

            # print('false caps: ', false_caps)
            image_names = error_dict["image"]
            for i, name in enumerate(image_names):
                image_obj = Image.open(os.path.join(self.challset_folder, name)).convert("RGB")
                # TODO: fix the llava-1.5-13b answers tonight and change the code to false_caps[i]
                false_cap = false_caps[0]
                ret_list.append((image_obj, name, true_cap, false_cap))
        
        return ret_list


    def probe_clip_emb(self, image_text_pairs) -> list:
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:5], "Checking CLIP embeddings: "):
            image = image_text_tup[0]
            text_caps = [image_text_tup[2], image_text_tup[3]] # [true, false]
            inputs = self.clip_processor(images=image, text=text_caps, return_tensors="pt", padding=True)
            outputs = self.clip(**inputs)
            scores = outputs.logits_per_image.squeeze()
            ret_tup = (image_text_tup[1], image_text_tup[2], image_text_tup[3], scores.tolist())
            scored_list.append(ret_tup)
        return scored_list


    def probe_llava_emb(self, image_text_pairs, pooling_type='max') -> list:
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:5], "Checking llava embeddings: "):
            image = image_text_tup[0]
            text_caps = [image_text_tup[2], image_text_tup[3]] # [true, false]
            visual_only_inputs = self.llava_processor(text="<image>", images=image, return_tensors="pt")
            text_only_inputs = self.llava_processor.tokenizer(text=text_caps, return_tensors="pt", padding=True)
            
            image_outputs = self.llava(**visual_only_inputs, output_hidden_states=True)
            text_outputs = self.llava(**text_only_inputs, output_hidden_states=True)

            image_embeds = image_outputs.hidden_states[-1]      # last hidden states
            text_embeds = text_outputs.hidden_states[-1]        # last hidden states

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

            # Compute cosine similarity via matrix multiplication
            logits_per_text = torch.matmul(pooled_text_embeds, pooled_image_embeds.t())  # Shape: (2, 1)
            scores = logits_per_text.squeeze()
            ret_tup = (image_text_tup[1], image_text_tup[2], image_text_tup[3], scores.tolist())
            scored_list.append(ret_tup)         
        return scored_list

    def __call__(self):
        input_list = self.prepare_input_pairs()
        clip_emb_list = self.probe_clip_emb(input_list)
        print("Stage 1 (CLIP) feature probing completed.")
        llava_emb_list = self.probe_llava_emb(input_list)
        print("Stage 2 (llava) feature probing completed.")

        # everything is fine, pack and go
        if len(clip_emb_list) == len(llava_emb_list):
            ret_list = []
            for i in range(len(clip_emb_list)):
                local_dict = {}
                clip_info_list = clip_emb_list[i]
                local_dict["image"] = clip_info_list[0]
                local_dict["true_cap"] = clip_info_list[1]
                local_dict["false_cap"] = clip_info_list[2]
                local_dict["clip_sim_score"] = clip_info_list[3]
                local_dict["llava_sim_score"] = llava_emb_list[i][3]
                ret_list.append(local_dict)
            output = os.path.join(self.challset_folder, 'feature_probing.json')
            write_json(output, ret_list)
        else:
            pass #TODO: finish the logic here
    

if __name__ == '__main__':
    config_path = 'llava-1.5-clip_config.json'
    probe_config = read_json(config_path)
    llava_probe = llava_feature_probe(probe_config)

    input_list = llava_probe.prepare_input_pairs()
    # clip_emb_list = llava_probe.probe_clip_emb(input_list)
    llava_emb_list = llava_probe.probe_llava_emb(input_list)

    # print(clip_emb_list[:5])
    print(llava_emb_list[:5])