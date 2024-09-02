import torch
from transformers import AutoProcessor, CLIPModel, LlavaForConditionalGeneration
from PIL import Image
import os, json
from tqdm import tqdm
import analyze_utils as utils

class llava_feature_analyzer:
    def __init__(self, config: dict) -> None:
        # this clip model should be modified to use -2 hidden layer's output as visual feature to align with llava setting
        # this modified clip model should NOT utilize logit_scale (originally set to e^2.6592) in order to only output cosine similarities      
        self.clip_path = config["clip_path"]
        self.llava_path = config["llava_path"]
        
        self.challset_folder = os.path.normpath(config['challset_folder'])
        self.index_file = config['index_file_name']
        self.prepared_inputs = config['prepared_inputs_name']
        self.dummy_cap = config['dummy_cap']
        self.action_exp = config['action_exp']
        self.place_exp = config['place_exp']
        self.dummy_exp = config['dummy_exp']
        


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
                false_cap = false_caps[i]
                ret_list.append((os.path.join(self.challset_folder, name), name, category, true_cap, false_cap, self.dummy_cap))

        prepared_file = os.path.join(self.challset_folder, self.prepared_inputs)
        utils.write_json(prepared_file, ret_list)
    
    
    def compare_clip_emb(self, image_text_pairs) -> list:
        clip_processor = AutoProcessor.from_pretrained(self.clip_path)
        clip = CLIPModel.from_pretrained(self.clip_path).to('cuda')
        clip = clip.eval()
        
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking CLIP embeddings: "):
            image_path = image_text_tup[0]
            image = utils.read_image(image_path)
            text_caps = [image_text_tup[3], image_text_tup[4], image_text_tup[5]]      # [true, false, dummy]
            
            inputs = clip_processor(images=image, text=text_caps, return_tensors="pt", padding=True)
            inputs = inputs.to(clip.device)
            with torch.no_grad():
                outputs = clip(**inputs)
            scores = outputs.logits_per_image.squeeze()
            ret_tup = image_text_tup[1:] + [scores.tolist()]
            scored_list.append(ret_tup)
        return scored_list


    def compare_adapter_emb(self, image_text_pairs, pooling_type='avg', use_prompt=True, contextualize_text=True) -> list:
        assert (use_prompt == contextualize_text), "If use PromptEOL strategy, contextualization is expected"
        llava_processor = AutoProcessor.from_pretrained(self.llava_path)
        llava = LlavaForConditionalGeneration.from_pretrained(self.llava_path, torch_dtype=torch.float16, device_map='auto').eval()
        
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking mmp embeddings: "):
            image_path = image_text_tup[0]
            category = image_text_tup[2]
            
            image = utils.read_image(image_path)
            if use_prompt:  
                text_caps = [cap.capitalize() + '.' for cap in image_text_tup[3:]]      # [true, false, dummy]
                prompt_eol = 'This sentence : "{sent_exp}" means in one word:"{word_exp}". This sentence : "{actual_sent}" means in one word:"'
                if category == 'action':
                    # text_query = [prompt_eol.format(sent_exp=self.action_exp[0], word_exp=self.action_exp[1], actual_sent=text_cap) for text_cap in text_caps]
                    text_query_tf = [prompt_eol.format(sent_exp=self.action_exp[0], word_exp=self.action_exp[1], actual_sent=text_cap) for text_cap in text_caps[:2]]
                    text_query_dm = [prompt_eol.format(sent_exp=self.dummy_exp[0], word_exp=self.dummy_exp[1], actual_sent=text_caps[2])]
                    text_query = text_query_tf + text_query_dm
                elif category == 'place':
                    # text_query = [prompt_eol.format(sent_exp=self.place_exp[0], word_exp=self.place_exp[1], actual_sent=text_cap) for text_cap in text_caps] 
                    text_query_tf = [prompt_eol.format(sent_exp=self.place_exp[0], word_exp=self.place_exp[1], actual_sent=text_cap) for text_cap in text_caps[:2]]
                    text_query_dm = [prompt_eol.format(sent_exp=self.dummy_exp[0], word_exp=self.dummy_exp[1], actual_sent=text_caps[2])]
                    text_query = text_query_tf + text_query_dm  
            else:
                text_query = image_text_tup[3:]
            
            image_inputs = llava_processor.image_processor(images=image, return_tensors="pt")
            llava_processor.tokenizer.padding_side = 'left'
            text_inputs = llava_processor.tokenizer(text=text_query, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                image_outputs = llava.vision_tower(image_inputs.pixel_values, output_hidden_states=True)
                # llava uses -2 hidden states as visual feature
                selected_image_feature = image_outputs.hidden_states[-2]
                # selected_image_feature = selected_image_feature[:, 1:]
                image_embeds = llava.multi_modal_projector(selected_image_feature)
                
                if contextualize_text:
                    text_outputs = llava(**text_inputs, output_hidden_states=True)
                    text_embeds = text_outputs.hidden_states[-1]
                else:
                    text_embeds = llava.language_model.get_input_embeddings()(text_inputs.input_ids)

            # Apply pooling to the embeddings
            pooled_image_embeds, pooled_text_embeds = utils.custom_pooling(image_embeds, text_embeds, pooling_type)
            cos_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)
            
            ret_tup = image_text_tup[1:] + [cos_sim.tolist()]
            scored_list.append(ret_tup)   
        return scored_list


    def compare_llm_emb(self, image_text_pairs, use_prompt=True, pooling_type='eol') -> list:
        llava_processor = AutoProcessor.from_pretrained(self.llava_path)
        llava = LlavaForConditionalGeneration.from_pretrained(self.llava_path, torch_dtype=torch.float16, device_map='auto').eval()

        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking llava embeddings: "):
            image_path = image_text_tup[0]
            category = image_text_tup[2]
            
            image = utils.read_image(image_path)
            if use_prompt:
                prompt_eol = 'This sentence : "{sent_exp}" means in one word:"{word_exp}". This sentence : "{actual_sent}" means in one word:"'
                text_caps = [cap.capitalize() + '.' for cap in image_text_tup[3:]]      # [true, false, dummy]
                img_symbol = "<image>"
                if category == 'action':
                    image_query_text = prompt_eol.format(sent_exp=self.action_exp[0], word_exp=self.action_exp[1], actual_sent=img_symbol)
                    text_query_tf = [prompt_eol.format(sent_exp=self.action_exp[0], word_exp=self.action_exp[1], actual_sent=text_cap) for text_cap in text_caps[:2]]
                    text_query_dm = [prompt_eol.format(sent_exp=self.dummy_exp[0], word_exp=self.dummy_exp[1], actual_sent=text_caps[2])]
                    text_query = text_query_tf + text_query_dm
                elif category == 'place':
                    image_query_text = prompt_eol.format(sent_exp=self.place_exp[0], word_exp=self.place_exp[1], actual_sent=img_symbol)
                    text_query_tf = [prompt_eol.format(sent_exp=self.place_exp[0], word_exp=self.place_exp[1], actual_sent=text_cap) for text_cap in text_caps[:2]]
                    text_query_dm = [prompt_eol.format(sent_exp=self.dummy_exp[0], word_exp=self.dummy_exp[1], actual_sent=text_caps[2])]
                    text_query = text_query_tf + text_query_dm 
            else:
                text_query = image_text_tup[3:]
                image_query_text = "<image>"
            
            visual_only_inputs = llava_processor(text=image_query_text, images=image, return_tensors="pt").to(llava.device)
            text_only_inputs = llava_processor.tokenizer(text=text_query, return_tensors="pt", padding=True).to(llava.device)
            
            with torch.no_grad():
                image_outputs = llava(**visual_only_inputs, output_hidden_states=True)
                text_outputs = llava(**text_only_inputs, output_hidden_states=True)

            image_embeds = image_outputs.hidden_states[-1]      # last hidden states
            text_embeds = text_outputs.hidden_states[-1]        # last hidden states

            # Apply pooling to the embeddings
            pooled_image_embeds, pooled_text_embeds = utils.custom_pooling(image_embeds, text_embeds, pooling_type) # Shape: (x, 5120)
            # Compute cosine similarity via matrix multiplication
            cos_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)
            ret_tup = image_text_tup[1:] + [cos_sim.tolist()]
            scored_list.append(ret_tup)         
        return scored_list


    def __call__(self):
        # self.prepare_input_pairs()
        input_list = utils.read_json(os.path.join(self.challset_folder, self.prepared_inputs))
        
        clip_emb_list = self.compare_clip_emb(input_list)
        # # print(clip_emb_list[:5])
        print(f"Stage 1 (CLIP) feature analysis completed.")
        torch.cuda.empty_cache()
        
        adapter_pooling_type = 'avg'
        adapter_emb_list = self.compare_adapter_emb(input_list, pooling_type=adapter_pooling_type, use_prompt=False, contextualize_text=True)
        print(f"Stage 2 (Adapter) feature analysis completed with {adapter_pooling_type} pooling.")
        # # print(adapter_emb_list[:5])
        torch.cuda.empty_cache()

        lm_pooling_type = 'eol'
        lm_emb_list = self.compare_llm_emb(input_list, pooling_type=lm_pooling_type)
        print(f"Stage 3 (LLaMA) feature analysis completed with {lm_pooling_type} pooling.")
        # print(lm_emb_list[:20])
        torch.cuda.empty_cache()
        
        # everything is fine, pack and go
        if len(clip_emb_list) == len(adapter_emb_list) == len(lm_emb_list):
            ret_list = []
            for i in range(len(clip_emb_list)):
                clip_info_list = clip_emb_list[i]
                local_dict = {"image": clip_info_list[0], "category": clip_info_list[1],  "true_cap": clip_info_list[2],
                              "false_cap": clip_info_list[3], "dummy_cap": clip_info_list[4],
                              "clip_sim_score": clip_info_list[5], "adapter_sim_score": adapter_emb_list[i][5], "lm_sim_score": lm_emb_list[i][5]}
                ret_list.append(local_dict)
            output = os.path.join(self.challset_folder, f'llava-1.5_image_text_sim_scores.json')
            utils.write_json(output, ret_list)
        # else:
        #     pass #TODO: finish the logic here
    

if __name__ == '__main__':
    config_path = 'llava-1.5-clip_analyzer_config.json'
    analyzer_config = utils.read_json(config_path)
    analyzer = llava_feature_analyzer(analyzer_config)
    # analyzer.prepare_input_pairs()
    analyzer()