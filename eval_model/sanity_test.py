import torch, json, os, copy
from transformers import AutoProcessor, LlavaForConditionalGeneration, InstructBlipForConditionalGeneration, InstructBlipProcessor, AutoModelForCausalLM, AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlamaTokenizer, LlavaNextImageProcessor
from PIL import Image
from templates import route_templates
from prompt import generate_mcq_prompt_st, generate_yn_prompt_st, generate_sa_prompt_st
from utils import read_json, write_json, infer_target_type
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from tqdm import tqdm
from transformers import logging as hf_logging
from eval import load_local_model
import torch.nn.functional as F



def process_input(model_type, processor, text_prompt, model_device):
    if model_type in ["llava-vicuna",  "llava-llama3", "llava-yi"]:
        inputs = processor.tokenizer(text=text_prompt, return_tensors="pt").to(model_device)
        input_ids = inputs.input_ids
    elif model_type in ["instructblip-t5", "blip2-t5"]:
        inputs = processor.tokenizer(text_prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
    elif model_type in ["qwen-vl", "qwen-vl-chat"]:
        processor.pad_token_id = processor.eod_id
        processor.padding_side = 'left'
        inputs = processor(text_prompt, return_tensors='pt', padding=True).to(model_device)
        input_ids = inputs.input_ids
    # elif model_type == :
    #     pass
        # inputs = processor.from_list_format([
        #     {'image': image_path},
        #     {'text': text_prompt},
        # ])
    else:
        raise ValueError("Unsupported model type")
    return inputs, input_ids


def model_generate(model, processor, model_type, inputs, max_new_tokens=10):
    if model_type != "qwen-vl-chat":  # use generate() method
        if model_type in ['blip2-t5', 'instructblip-t5']:
            generate_model = model.language_model
        else:
            generate_model = model
        
        inputs = inputs.to(generate_model.device)
        with torch.no_grad():
            output_dict = generate_model.generate(**inputs, do_sample=False, return_dict_in_generate=True, 
                                        output_logits=True, max_new_tokens=max_new_tokens)
        output_logits = output_dict.logits
        output_text = processor.decode(output_dict.sequences[0], skip_special_tokens=True).strip() # batch_size = 1
        output_ids = output_dict.sequences
        
        # parse the output_text
        if model_type == "llava-vicuna":
            output_text = output_text.split('ASSISTANT: ')[1]
        elif model_type == "qwen-vl":
            attempt = output_text.split('Answer:')[1].split('\n')[0].strip()
            if attempt != '':
                output_text = attempt
        elif model_type in ["llava-llama3", "llava-yi"]:
            output_text = output_text.split("assistant")[-1].strip()
    
    else:
        _, _, output_dict = model.chat(processor, query=inputs.input_ids[0].tolist(), history=None) # here need to change modeling_qwen.py
        output_logits = output_dict.logits
        output_text = processor.decode(output_dict.sequences[0], skip_special_tokens=True).split("assistant")[-1].strip()
        output_ids = output_dict.sequences
        
    return output_text, output_logits, output_ids


def compute_uncertainty_from_logits(logits, mask=None, epsilon=1e-6):
    logits_tensor = torch.cat([t for t in logits], dim=0)
    probs = F.softmax(logits_tensor, dim=-1)  # Shape: (batch_size, sequence_length, vocab_size)
    # Compute entropy for each token
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)  # Shape: (batch_size, sequence_length)
    if mask:
        masked_entropy = entropy * mask
        mean_entropy = masked_entropy.sum() / mask.sum()
    else:
        mean_entropy = entropy.mean()
    return mean_entropy.item()


def qwen_vl_chat_context_mask(tokenizer, output_ids, token_1='<|im_start|>', token_2=b'assistant'):
    token_id_1 = tokenizer.convert_tokens_to_ids(token_1)
    token_id_2 = tokenizer.convert_tokens_to_ids(token_2)
    
    batch_size, sequence_length = output_ids.shape
    mask = torch.ones_like(output_ids)
    found = False
    for i in range(batch_size):
        for j in range(sequence_length - 1):
            if output_ids[i, j] == token_id_1 and output_ids[i, j + 1] == token_id_2:
                found = True
                mask[i, :j + 2] = 0
                break
        if not found:
            print(f"Response token not found in sequence")
    return mask


# def construct_context_mask(input_ids, output_ids, model_type):
#     if model_type in ["llava-vicuna", "llava-llama3", "llava-yi", "qwen-vl"]:
#         input_ids_mask = torch.ones_like(output_ids)
#         # Ensure the front part of output_ids is identical to input_ids
#         input_length = input_ids.shape[1]
#         assert torch.equal(output_ids[:, :input_length], input_ids), "The front part of output_ids must be identical to input_ids"
#         # Mask the input_ids part
#         input_ids_mask[:, :input_length] = 0
#         return input_ids_mask           
#     else:
#         raise NotImplementedError("This function is not implemented for the model type")
        


def sanity_test(model_dir, model_type, text_input_path, text_output_path, 
               prefix='', postfix='', use_device_map=True, max_new_tokens=25):
    model, processor = load_local_model(model_dir=model_dir, model_type=model_type, use_device_map=use_device_map)
    input_list_of_dict = read_json(text_input_path)
    output_list_of_dict = []
    with open(text_output_path, "w") as f:
        f.write("[\n")
        for idx, input_dict in tqdm(enumerate(input_list_of_dict[:]), total=len(input_list_of_dict), desc="Infering answers"):
            input_type = infer_target_type(input_dict["image"][0])
            image_names = input_dict["image"]

            subject = input_dict["context"]["subject"]
            if input_type == "action":
                action = input_dict["target"]["action"]
                place = input_dict["context"]["place"]
            elif input_type == "place":
                action = input_dict["context"]["action"]
                place = input_dict["target"]["place"]
            gt_triplet = [subject, action, place]
            model_templates = route_templates(model_type=model_type, sanity_test=True, use_cot=False)
            mcq = generate_mcq_prompt_st(gt_triplet=gt_triplet, mcq_dict=input_dict["MCQ_options"], target=input_type, 
                                    mcq_prompt_template=model_templates["MCQ"], postfix=postfix)
            yn = generate_yn_prompt_st(gt_triplet=gt_triplet, yn_prompt_template=model_templates["YN"], target=input_type, 
                                    prefix=prefix, postfix=postfix)
            sa = generate_sa_prompt_st(gt_triplet=gt_triplet, sa_prompt_template=model_templates["SA"], target=input_type, 
                                    prefix=prefix, postfix=postfix)
            prompt_dict = {"mcq": mcq, "yn": yn, "sa": sa}
            output_dict = copy.deepcopy(input_dict)
            output_dict["category"] = input_type
            for q_type, prompt in prompt_dict.items():
                print('prompt:', prompt)
                output_dict[q_type] = prompt
                # print('prompt:', prompt)
                output_dict[f"{q_type}_model_ans"] = []
                for repeat in image_names:
                    inputs, input_ids = process_input(model_type=model_type, processor=processor, text_prompt=prompt, model_device=model.device)
                    
                    output_text, output_logits, output_ids = model_generate(model=model, processor=processor, model_type=model_type, 
                                                inputs=inputs, max_new_tokens=max_new_tokens)
                    
                    # context_mask = construct_context_mask(input_ids=input_ids, output_ids=output_ids, model_type=model_type)
                    # context_mask = qwen_vl_chat_context_mask(tokenizer=processor, output_ids=output_ids)   
                    # # uncertainty = compute_uncertainty(model=model, generate_ids=output_ids, input_ids_mask=context_mask)
                    # uncertainty = compute_uncertainty(model=model, generate_ids=output_ids, input_ids_mask=context_mask)
                    print(f"Length of output_logits: {len(output_logits)}")
                    uncertainty = compute_uncertainty_from_logits(logits=output_logits)
                    
                    print(output_text)
                    print('uncertainty: ', uncertainty)
                    output_dict[f"{q_type}_model_ans"].append([output_text])
                    output_dict[f"{q_type}_model_ans_uncertainty"] = uncertainty
            f.write(json.dumps(output_dict, indent=4))
            if idx != len(input_list_of_dict) - 1:
                f.write(",\n")
            f.flush()
            output_list_of_dict.append(output_dict)

        f.write('\n]')
        f.flush()        
    print(f"Finished on all the {len(output_list_of_dict)} inputs")
    # write_json(json_list=output_list_of_dict, json_path=text_output_path)
    return output_list_of_dict
                    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    text_input = "/120040051/test_resource/merged0728/input_info.json"
    model_type = "instructblip-t5"
    model_dir = "/120040051/MLLM_Repos/instructblip-flan-t5-xxl"
    
    model_name = os.path.basename(model_dir)
    output_folder = "/120040051/Github_Repos/VKConflict/eval_model/sanity_test_results"
    text_output = os.path.join(output_folder, f"{model_name}_with_uncertainty_new.json")

    hf_logging.set_verbosity_error()
    sanity_test(model_dir=model_dir, model_type=model_type, text_input_path=text_input, 
               text_output_path=text_output, max_new_tokens=10, use_device_map=True)