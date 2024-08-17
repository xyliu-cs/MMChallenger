import torch
from transformers import AutoProcessor, CLIPModel, LlavaForConditionalGeneration
from PIL import Image
import os, json
from tqdm import tqdm
from ..utils import read_json, write_json

class llava_feature_probe:
    def __init__(self, config: dict) -> None:
        # this clip model should be modified to use -2 hidden layer's output as visual feature to align with llava setting
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
            category = error_list["category"]
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

            image_names = error_dict["image"]
            for i, name in enumerate(image_names):
                image_obj = Image.open(os.path.join(self.challset_folder, name)).convert("RGB")
                false_cap = false_caps[i]
                ret_list.append((image_obj, name, true_cap, false_cap))
        
        return ret_list


    def probe_clip_emb(self, image_text_pairs) -> list:
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs, "Checking CLIP embeddings: "):
            image = image_text_tup[0]
            text_caps = [image_text_tup[2], image_text_tup[3]] # [true, false]
            inputs = self.clip_processor(images=image, text=text_caps, return_tensors="pt", padding=True)
            outputs = self.clip(**inputs)
            scores = outputs.logits_per_image
            ret_tup = (image_text_tup[1], image_text_pairs[2], image_text_pairs[3], scores.tolist())
            scored_list.append(ret_tup)
        return scored_list


    def probe_llava_emb(self, image_text_pairs, logit_scale=1.0, pooling_type='max') -> list:
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs, "Checking llava embeddings: "):
            image = image_text_tup[0]
            text_caps = [image_text_tup[2], image_text_tup[3]] # [true, false]
            visual_only_inputs = self.llava_processor(text="<image>", images=image, return_tensors="pt")
            text_only_inputs = self.llava_processor.tokenizer(text=text_caps, return_tensors="pt", padding=True)
            
            image_embeds = self.llava(**visual_only_inputs, output_hidden_states=True)
            text_embeds = self.llava(**text_only_inputs, output_hidden_states=True)

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
            logits_per_text = torch.matmul(pooled_text_embeds, pooled_image_embeds.t()) * logit_scale  # Shape: (2, 1)
            scores = logits_per_text.squeeze()
            ret_tup = (image_text_tup[1], image_text_pairs[2], image_text_pairs[3], scores.tolist())
            scored_list.append(ret_tup)         
            return scored_list
    

if __name__ == '__main__':
    config_path = 'llava-1.5-clip_config.json'
    probe_config = read_json(config_path)
    llava_probe = llava_feature_probe(probe_config)

    input_list = llava_probe.prepare_input_pairs()
    clip_emb_list = llava_probe.probe_clip_emb(input_list)
    llava_emb_list = llava_probe.probe_llava_emb(input_list)

    print(clip_emb_list[:5])
    print(llava_emb_list[:5])
