import torch
from transformers import AutoModelForCausalLM, CLIPModel, AutoTokenizer
from PIL import Image
import os, json
from tqdm import tqdm
import analyze_utils as utils
import open_clip


class qwen_vl_feature_analyzer:
    def __init__(self, config: dict) -> None:
        # this clip model should be modified to use -2 hidden layer's output as visual feature to align with blip-2 setting
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.challset_folder = os.path.normpath(config['challset_folder'])
        self.index_file = config['index_file_name']
        self.prepared_inputs = config['prepared_inputs_name']
        self.dummy_cap = config['dummy_cap']
        
        self.open_clip = config['open_clip_path']
        self.qwen_vl_path = config['qwen_vl_path']
        
        self.action_exp = config['action_exp']
        self.place_exp = config['place_exp']

        # self.vision_features = []
        # self.adaptor_features = []
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
                ret_list.append((os.path.join(self.challset_folder, name), name, true_cap, false_cap, self.dummy_cap))

        prepared_file = os.path.join(self.challset_folder, self.prepared_inputs)
        utils.write_json(prepared_file, ret_list)    
    
    # TODO: Need to change the visual feature to use from last to -2
    def compare_clip_emb(self, image_text_pairs, pooling_type) -> list:
        
        openclip = CLIPModel.from_pretrained(self.open_clip).to("cuda")
        openclip_tokenizer = AutoTokenizer.from_pretrained(self.open_clip)
        qwen_vl = AutoModelForCausalLM.from_pretrained(self.qwen_vl_path, trust_remote_code=True, device_map='auto', fp16=True)
        qwen_pcr = AutoTokenizer.from_pretrained(self.qwen_vl_path, trust_remote_code=True)
        
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking CLIP embeddings: "):
            text_caps = [image_text_tup[2], image_text_tup[3], image_text_tup[4]]      # [true, false, dummy]
            # query = qwen_pcr.from_list_format([
            #     {'image': image_text_tup[0]},
            # ])
            # img_inputs = qwen_pcr(query, return_tensors='pt').to(qwen_vl.device)
            text_inputs = openclip_tokenizer(text_caps, padding=True, return_tensors="pt").to(openclip.device)
            with torch.no_grad():
                _, raw_image_features = qwen_vl.transformer.visual.encode([image_text_tup[0]])
                # using the pre-trained projection layer 
                raw_image_features = raw_image_features.float()
                image_features = openclip.visual_projection(raw_image_features.to(openclip.device))
                # Apply pooling to the embeddings
                if pooling_type == 'tok':
                    # [cls]
                    pooled_image_embeds = image_features[:, 0, :]
                    # [eos]
                    pooled_text_embeds = openclip.get_text_features(**text_inputs)
                elif pooling_type == 'avg':
                    text_outputs = openclip.text_model(**text_inputs, output_hidden_states=True)
                    text_feature = text_outputs.last_hidden_state
                    pooled_image_embeds = torch.mean(image_features, dim=1)  
                    pooled_text_embeds = torch.mean(text_feature, dim=1) 
                else:
                    raise ValueError("Invalid pooling type. Use 'max' or 'avg'.")                
                
            cosine_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)
            # Append pooled features
            # self.vision_features.append((image_features, text_features))
            ret_tup = (image_text_tup[1], image_text_tup[2], image_text_tup[3], image_text_tup[4], cosine_sim.tolist())
            scored_list.append(ret_tup)
        return scored_list


    def compare_adapter_emb(self, image_text_pairs, pooling_type='avg') -> list:
        
        qwen_vl = AutoModelForCausalLM.from_pretrained(self.qwen_vl_path, trust_remote_code=True, fp16=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(self.qwen_vl_path, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.padding_side = 'right'
        
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking adapter embeddings: "):
            text_caps = [image_text_tup[2], image_text_tup[3], image_text_tup[4]]      # [true, false, dummy]

            # img_query = tokenizer.from_list_format([
            #     {'image': image_text_tup[0]},
            # ])
            # img_inputs = tokenizer(img_query, return_tensors='pt').to(qwen_vl.device)
            
            text_inputs = tokenizer(text_caps, padding=True, return_tensors="pt").to(qwen_vl.device)
            # text_input_ids = text_inputs.input_ids
            with torch.no_grad():
                adapter_img_features, _ = qwen_vl.transformer.visual.encode([image_text_tup[0]])
                text_features = qwen_vl.transformer.get_input_embeddings()(text_inputs.input_ids)
                # Apply pooling to the embeddings
                if pooling_type == 'avg':
                    pooled_image_embeds = torch.mean(adapter_img_features, dim=1)  
                    pooled_text_embeds = torch.mean(text_features, dim=1)
                elif pooling_type == 'max':
                    pooled_image_embeds = torch.max(adapter_img_features, dim=1)[0]  
                    pooled_text_embeds = torch.max(text_features, dim=1)[0]
                else:
                    raise ValueError("Invalid pooling type. Use 'max' or 'avg'.")                
                
            cosine_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)
            # Append pooled features
            # self.vision_features.append((image_features, text_features))
            ret_tup = (image_text_tup[1], image_text_tup[2], image_text_tup[3], image_text_tup[4], cosine_sim.tolist())
            scored_list.append(ret_tup)
        return scored_list


    def compare_qwen_emb(self, image_text_pairs, pooling_type='avg') -> list:
        
        qwen_vl = AutoModelForCausalLM.from_pretrained(self.qwen_vl_path, trust_remote_code=True, fp16=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(self.qwen_vl_path, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.padding_side = 'left'
        
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking qwen embeddings: "):
            text_caps = [cap.capitalize() + '.' for cap in image_text_tup[2:]]      # [true, false, dummy]

            # TODO: add some logic to distinguish the question type
            prompt_eol = 'This sentence : "{sent_exp}" means in one word:"{word_exp}". This sentence : "{actual_sent}" means in one word:"'
            img_seq = f"<img>{image_text_tup[0]}</img>"
            
            if image_text_tup['type'] == 'action':
                image_query = prompt_eol.format(sent_exp=self.action_exp[0], word_exp=self.action_exp[1], actual_sent=img_seq)
                text_query = [prompt_eol.format(sent_exp=self.action_exp[0], word_exp=self.action_exp[1], actual_sent=text_cap) for text_cap in text_caps]
            elif image_text_tup['type'] == 'place':
                image_query = prompt_eol.format(sent_exp=self.place_exp[0], word_exp=self.place_exp[1], actual_sent=img_seq)
                text_query = [prompt_eol.format(sent_exp=self.place_exp[0], word_exp=self.place_exp[1], actual_sent=text_cap) for text_cap in text_caps]  
            
            img_inputs = tokenizer(image_query, return_tensors='pt').to(qwen_vl.device)
            text_inputs = tokenizer(text_query, padding=True, return_tensors="pt").to(qwen_vl.device)
            
            # text_input_ids = text_inputs.input_ids
            with torch.no_grad():
                qwen_img_features = qwen_vl(**img_inputs, output_hidden_states=True)
                qwen_text_features = qwen_vl(**text_inputs, output_hidden_states=True)
                img_lhs = qwen_img_features.hidden_states[-1]
                text_lhs = qwen_text_features.hidden_states[-1]

            # Apply pooling to the embeddings
            if pooling_type == 'avg':
                pooled_image_embeds = torch.mean(img_lhs, dim=1)  
                pooled_text_embeds = torch.mean(text_lhs, dim=1)
            elif pooling_type == 'max':
                pooled_image_embeds = torch.max(img_lhs, dim=1)[0]  
                pooled_text_embeds = torch.max(text_lhs, dim=1)[0]
            elif pooling_type == 'eol':
                pooled_image_embeds = img_lhs[:, -1, :]
                # text is right aligned since we pad to the left
                pooled_text_embeds = text_lhs[:, -1, :] 
            else:
                raise ValueError("Invalid pooling type. Use 'max' or 'avg'.")                
                
            cosine_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)
            # Append pooled features
            # self.vision_features.append((image_features, text_features))
            ret_tup = (image_text_tup[1], image_text_tup[2], image_text_tup[3], image_text_tup[4], cosine_sim.tolist())
            scored_list.append(ret_tup)        
        return scored_list

    def __call__(self):
        # self.prepare_input_pairs()
        input_list = utils.read_json(os.path.join(self.challset_folder, self.prepared_inputs))
        # clip_emb_list = self.compare_clip_emb(input_list, pooling_type='tok')
        # # print(clip_emb_list[:5])
        # print("Stage 1 (CLIP) feature analysis completed.")
        # torch.cuda.empty_cache()
        
        pooling_type = 'eol'
        # adapter_emb_list = self.compare_adapter_emb(input_list, pooling_type=pooling_type)
        # print(f"Stage 2 (Adapter) feature analysis completed with {pooling_type} pooling.")
        # # print(adapter_emb_list[:5])
        # torch.cuda.empty_cache()

        qwen_emb_list = self.compare_qwen_emb(input_list, pooling_type=pooling_type)
        print(f"Stage 3 (Qwen) feature analysis completed with {pooling_type} pooling.")
        print(qwen_emb_list[:20])
        # torch.cuda.empty_cache()
        # # everything is fine, pack and go
        # if len(clip_emb_list) == len(adapter_emb_list) == len(qwen_emb_list):
        #     ret_list = []
        #     for i in range(len(clip_emb_list)):
        #         local_dict = {}
        #         clip_info_list = clip_emb_list[i]
        #         local_dict["image"] = clip_info_list[0]
        #         local_dict["true_cap"] = clip_info_list[1]
        #         local_dict["false_cap"] = clip_info_list[2]
        #         local_dict["dummy_cap"] = clip_info_list[3]

        #         local_dict["clip_sim_score"] = clip_info_list[4]
        #         local_dict[f"adapter_sim_score_{pooling_type}"] = adapter_emb_list[i][4]
        #         local_dict[f"qwen_sim_score_{pooling_type}"] = qwen_emb_list[i][4]
                
        #         ret_list.append(local_dict)
        #     output = os.path.join(self.challset_folder, f'qwen_vl_feature_sim_score_{pooling_type}_pooling.json')
        #     utils.write_json(output, ret_list)
        # else:
        #     pass #TODO: finish the logic here
    

if __name__ == '__main__':
    config_path = 'qwen-vl-openclip_analyzer_config.json'
    analyzer_config = utils.read_json(config_path)
    analyzer = qwen_vl_feature_analyzer(analyzer_config)
    # analyzer.prepare_input_pairs()

    analyzer()