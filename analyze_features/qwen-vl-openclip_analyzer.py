import torch
from transformers import AutoModelForCausalLM, CLIPModel, AutoTokenizer
from PIL import Image
import os, json
from tqdm import tqdm
import analyze_utils as utils


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
        self.dummy_exp = config['dummy_exp']

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
                ret_list.append((os.path.join(self.challset_folder, name), name, category, true_cap, false_cap, self.dummy_cap))

        prepared_file = os.path.join(self.challset_folder, self.prepared_inputs)
        utils.write_json(prepared_file, ret_list)    


    # TODO: Need to change the visual feature to use from last to -2
    def compare_clip_emb(self, image_text_pairs, pooling_type) -> list:
        assert pooling_type in ['tok', 'avg', 'max']
        openclip = CLIPModel.from_pretrained(self.open_clip).to("cuda")
        openclip_tokenizer = AutoTokenizer.from_pretrained(self.open_clip)
        qwen_vl = AutoModelForCausalLM.from_pretrained(self.qwen_vl_path, trust_remote_code=True, device_map='auto', fp16=True)
        # qwen_pcr = AutoTokenizer.from_pretrained(self.qwen_vl_path, trust_remote_code=True)
        
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking CLIP embeddings: "):
            text_caps = image_text_tup[3:]      # [true, false, dummy]
            image_path  = image_text_tup[0]
            text_inputs = openclip_tokenizer(text_caps, padding=True, return_tensors="pt").to(openclip.device)
            
            with torch.no_grad():
                _, raw_image_features = qwen_vl.transformer.visual.encode([image_path])
                # using the pre-trained projection layer 
                raw_image_features = raw_image_features.float()
                image_features = openclip.visual_projection(raw_image_features.to(openclip.device))
                # Apply pooling to the embeddings
                if pooling_type == 'tok':
                    # [cls]
                    pooled_image_embeds = image_features[:, 0, :]
                    # [eos]
                    pooled_text_embeds = openclip.get_text_features(**text_inputs)
                else:
                    text_outputs = openclip.text_model(**text_inputs, output_hidden_states=True)
                    text_features = text_outputs.last_hidden_state
                    pooled_image_embeds, pooled_text_embeds = utils.custom_pooling(image_features, text_features, pooling_type)
            
            cosine_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)
            # Append pooled features
            # self.vision_features.append((image_features, text_features))
            ret_tup = image_text_tup[1:] + [cosine_sim.tolist()]
            scored_list.append(ret_tup)
        return scored_list


    def compare_adapter_emb(self, image_text_pairs, pooling_type='avg', use_prompt=True, contextualize_text=True) -> list:
        assert (use_prompt == contextualize_text), "If use PromptEOL strategy, contextualization is expected"
        assert pooling_type in ['avg', 'max', 'avg_eol']
        qwen_vl = AutoModelForCausalLM.from_pretrained(self.qwen_vl_path, trust_remote_code=True, fp16=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(self.qwen_vl_path, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.padding_side = 'left'
        
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking adapter embeddings: "):
            category = image_text_tup[2]
            image_path = image_text_tup[0]
            
            # use prompteol for the text
            if use_prompt:  
                text_caps = [cap.capitalize() + '.' for cap in image_text_tup[3:]]      # [true, false, dummy]
                prompt_eol = 'This sentence : "{sent_exp}" means in one word:"{word_exp}". This sentence : "{actual_sent}" means in one word:"'
                if category == 'action':
                    text_query_tf = [prompt_eol.format(sent_exp=self.action_exp[0], word_exp=self.action_exp[1], actual_sent=text_cap) for text_cap in text_caps[:2]]
                    text_query_dm = [prompt_eol.format(sent_exp=self.dummy_exp[0], word_exp=self.dummy_exp[1], actual_sent=text_caps[2])]
                    text_query = text_query_tf + text_query_dm
                elif category == 'place':
                    text_query_tf = [prompt_eol.format(sent_exp=self.place_exp[0], word_exp=self.place_exp[1], actual_sent=text_cap) for text_cap in text_caps[:2]]
                    text_query_dm = [prompt_eol.format(sent_exp=self.dummy_exp[0], word_exp=self.dummy_exp[1], actual_sent=text_caps[2])]
                    text_query = text_query_tf + text_query_dm
            else:
                text_query = image_text_tup[3:]
                
            # print('text query is: ', text_query)
            text_inputs = tokenizer(text_query, padding=True, return_tensors="pt").to(qwen_vl.device)
            
            with torch.no_grad():
                adapter_img_features, _ = qwen_vl.transformer.visual.encode([image_path])
                # print('adater image feature shape: ', adapter_img_features.shape)  # (1, 256, 4096)
                if contextualize_text:
                    text_output = qwen_vl(**text_inputs, output_hidden_states=True)
                    text_features = text_output.hidden_states[-1]
                else:
                    text_features = qwen_vl.transformer.get_input_embeddings()(text_inputs.input_ids)
    
            # Apply pooling to the embeddings
            pooled_image_embeds, pooled_text_embeds = utils.custom_pooling(adapter_img_features, text_features, pooling_type)
            cosine_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)
            
            # Append pooled features
            # self.vision_features.append((image_features, text_features))
            ret_tup = image_text_tup[1:] + [cosine_sim.tolist()]
            scored_list.append(ret_tup)
        return scored_list


    def compare_llm_emb(self, image_text_pairs, use_prompt=True, pooling_type='eol') -> list:  
        qwen_vl = AutoModelForCausalLM.from_pretrained(self.qwen_vl_path, trust_remote_code=True, fp16=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(self.qwen_vl_path, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.padding_side = 'left'
        
        scored_list = []
        for image_text_tup in tqdm(image_text_pairs[:], "Checking qwen embeddings: "):
            image_path = image_text_tup[0]
            category = image_text_tup[2]
            
            if use_prompt:
                prompt_eol = 'This sentence : "{sent_exp}" means in one word:"{word_exp}". This sentence : "{actual_sent}" means in one word:"'
                text_caps = [cap.capitalize() + '.' for cap in image_text_tup[3:]]      # [true, false, dummy]
                img_seq = f"<img>{image_path}</img>"
                if category == 'action':
                    image_query = prompt_eol.format(sent_exp=self.action_exp[0], word_exp=self.action_exp[1], actual_sent=img_seq)
                    text_query_tf = [prompt_eol.format(sent_exp=self.action_exp[0], word_exp=self.action_exp[1], actual_sent=text_cap) for text_cap in text_caps[:2]]
                    text_query_dm = [prompt_eol.format(sent_exp=self.dummy_exp[0], word_exp=self.dummy_exp[1], actual_sent=text_caps[2])]
                    text_query = text_query_tf + text_query_dm
                elif category == 'place':
                    image_query = prompt_eol.format(sent_exp=self.place_exp[0], word_exp=self.place_exp[1], actual_sent=img_seq)
                    text_query_tf = [prompt_eol.format(sent_exp=self.place_exp[0], word_exp=self.place_exp[1], actual_sent=text_cap) for text_cap in text_caps[:2]]
                    text_query_dm = [prompt_eol.format(sent_exp=self.dummy_exp[0], word_exp=self.dummy_exp[1], actual_sent=text_caps[2])]
                    text_query = text_query_tf + text_query_dm
            else:
                text_query = image_text_tup[3:]
                image_query = f"<img>{image_path}</img>"
            # print(image_query)
            # print(text_query)
            
            img_inputs = tokenizer(image_query, return_tensors='pt').to(qwen_vl.device)
            text_inputs = tokenizer(text_query, padding=True, return_tensors="pt").to(qwen_vl.device)
            
            # text_input_ids = text_inputs.input_ids
            with torch.no_grad():
                qwen_img_features = qwen_vl(**img_inputs, output_hidden_states=True)
                qwen_text_features = qwen_vl(**text_inputs, output_hidden_states=True)
                img_lhs = qwen_img_features.hidden_states[-1]
                text_lhs = qwen_text_features.hidden_states[-1]

            # Apply pooling to the embeddings
            pooled_image_embeds, pooled_text_embeds = utils.custom_pooling(img_lhs, text_lhs, pooling_type)             
            cosine_sim = utils.cosine_sim(pooled_image_embeds, pooled_text_embeds)
            # Append pooled features
            # self.vision_features.append((image_features, text_features))
            ret_tup = image_text_tup[1:] + [cosine_sim.tolist()]
            scored_list.append(ret_tup)        
        return scored_list


    def __call__(self):
        # self.prepare_input_pairs()
        input_list = utils.read_json(os.path.join(self.challset_folder, self.prepared_inputs))
        vision_model_pooling_type = 'tok'
        clip_emb_list = self.compare_clip_emb(input_list, pooling_type=vision_model_pooling_type)
        # # print(clip_emb_list[:5])
        print(f"Stage 1 (CLIP) feature analysis completed with {vision_model_pooling_type} pooling.")
        torch.cuda.empty_cache()
        
        adapter_pooling_type = 'avg_eol' # or 'avg'
        adapter_emb_list = self.compare_adapter_emb(input_list, pooling_type=adapter_pooling_type, use_prompt=True, contextualize_text=True)
        print(f"Stage 2 (Adapter) feature analysis completed with {adapter_pooling_type} pooling.")
        # # print(adapter_emb_list[:5])
        torch.cuda.empty_cache()

        llm_pooling_type = 'eol'
        llm_emb_list = self.compare_llm_emb(input_list, use_prompt=True, pooling_type=llm_pooling_type)
        print(f"Stage 3 (LLM) feature analysis completed with {llm_pooling_type} pooling.")
        
        # output = os.path.join(self.challset_folder, f'qwen_vl_feature_sim_score.json')
        # utils.write_json(output, qwen_emb_list)
        # print(qwen_emb_list[:20])
        torch.cuda.empty_cache()
        # everything is fine, pack and go
        if len(clip_emb_list) == len(adapter_emb_list) == len(llm_emb_list):
            ret_list = []
            for i in range(len(clip_emb_list)):
                clip_info_list = clip_emb_list[i]
                local_dict = {"image": clip_info_list[0], "category": clip_info_list[1],  "true_cap": clip_info_list[2],
                              "false_cap": clip_info_list[3], "dummy_cap": clip_info_list[4],
                              "clip_sim_score": clip_info_list[5], "adapter_sim_score": adapter_emb_list[i][5], "lm_sim_score": llm_emb_list[i][5]}
                ret_list.append(local_dict)
            output = os.path.join(self.challset_folder, f'qwen_vl_image_text_sim_scores.json')
            utils.write_json(output, ret_list)
        # else:
        #     pass #TODO: finish the logic here
    

if __name__ == '__main__':
    config_path = 'qwen-vl-openclip_analyzer_config.json'
    analyzer_config = utils.read_json(config_path)
    analyzer = qwen_vl_feature_analyzer(analyzer_config)
    # analyzer.prepare_input_pairs()

    analyzer()