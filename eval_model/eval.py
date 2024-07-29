import torch, json, os, copy
from transformers import AutoProcessor, LlavaForConditionalGeneration, InstructBlipForConditionalGeneration, InstructBlipProcessor
from PIL import Image
from templates import route_templates
from prompt import generate_mcq_prompt, generate_yn_prompt, generate_sa_prompt
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from tqdm import tqdm
from transformers import logging as hf_logging


def load_local_model(model_dir, model_type, use_device_map=True, device_map='auto', use_half_precision=False):
    assert model_type in ["llava-vicuna", "instructblip"], f"Unsupported model type {model_type}"
    torch.cuda.empty_cache()
    if model_type == "llava-vicuna":
        processor = AutoProcessor.from_pretrained(model_dir)
        if use_device_map:
            print(f"Setting device map to {device_map}")
            model = LlavaForConditionalGeneration.from_pretrained(model_dir, device_map=device_map)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device to {device}")
            model = LlavaForConditionalGeneration.from_pretrained(model_dir).to(device)
    elif model_type == "instructblip":
        processor = InstructBlipProcessor.from_pretrained(model_dir)
        if use_device_map:
            with init_empty_weights():
                instblip = InstructBlipForConditionalGeneration.from_pretrained(model_dir)
                dmap = infer_auto_device_map(instblip, max_memory={0: "20GiB", 1: "20GiB"}, no_split_module_classes=['InstructBlipEncoderLayer', 'InstructBlipQFormerLayer', 'LlamaDecoderLayer'])
                dmap['language_model.lm_head'] = dmap['language_projection'] = dmap[('language_model.model.embed_tokens')]
            model = load_checkpoint_and_dispatch(instblip, model_dir, device_map=dmap)
        else:
            model = InstructBlipForConditionalGeneration.from_pretrained(model_dir).to(device)
    return model.eval(), processor


def read_json(json_path):
    print(f"Read json file from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data[:]


def write_json(json_path, json_list):
    with open(json_path, 'w') as f:
        json.dump(json_list, f, indent=2)
    print(f"Write {len(json_list)} items to {json_path}")


def infer_target_type(image_name):
    if image_name.startswith("A"):
        target = "action"
    elif image_name.startswith("P"):
        target = "place"
    else:
        raise ValueError(f"Illegal image naming {image_name}")
    return target



def eval_model(model_dir, model_type, text_input_path, image_folder, text_output_path, prefix='', postfix='',use_device_map=True, half_prec=False):
    model, processor = load_local_model(model_dir=model_dir, model_type=model_type, use_device_map=use_device_map)
    input_list_of_dict = read_json(text_input_path)
    output_list_of_dict = []
    for input_dict in tqdm(input_list_of_dict, "Infering answers"):
        input_type = infer_target_type(input_dict["image"][0])
        image_names = input_dict["image"]
        image_paths = [os.path.join(image_folder, name) for name in image_names]
        subject = input_dict["context"]["subject"]
        if input_type == "action":
            action = input_dict["target"]["action"]
            place = input_dict["context"]["place"]
        elif input_type == "place":
            action = input_dict["context"]["action"]
            place = input_dict["target"]["place"]
        gt_triplet = [subject, action, place]
        model_templates = route_templates(model_type=model_type)
        mcq = generate_mcq_prompt(gt_triplet=gt_triplet, mcq_dict=input_dict["MCQ_options"], target=input_type, mcq_prompt_template=model_templates["MCQ"], postfix=postfix)
        yn = generate_yn_prompt(gt_triplet=gt_triplet, yn_prompt_template=model_templates["YN"], target=input_type, prefix=prefix, postfix=postfix)
        sa = generate_sa_prompt(gt_triplet=gt_triplet, sa_prompt_template=model_templates["SA"], target=input_type, prefix=prefix, postfix=postfix)
        prompt_dict = {"mcq": mcq, "yn": yn, "sa": sa}
        output_dict = copy.deepcopy(input_dict)
        output_dict["category"] = input_type
        for q_type, prompt in prompt_dict.items():
            output_dict[q_type] = prompt
            output_dict[f"{q_type}_model_ans"] = []
            for i, image_path in enumerate(image_paths):
                local_ans = []
                for iter in range(3):
                    raw_image = Image.open(image_path).convert("RGB")
                    inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)
                    output = model.generate(**inputs, max_new_tokens=25)[0]
                    output_text = processor.decode(output, skip_special_tokens=True).strip()
                    if model_type == "llava-vicuna":
                        output_text = output_text.split('ASSISTANT: ')[1]
                    local_ans.append(output_text)
                output_dict[f"{q_type}_model_ans"].append(local_ans)
        output_list_of_dict.append(output_dict)
    print(f"Finished on all the {len(output_list_of_dict)} inputs")
    write_json(json_list=output_list_of_dict, json_path=text_output_path)
                    

if __name__ == "__main__":
    text_input = "/120040051/merged0728/input_info.json"
    text_output = "/120040051/test_resource/output_answers/llava15_13b_outputs_common_sense.json"
    model_type = "llava-vicuna"
    model_dir = "/120040051/llava-1.5-13b-hf"
    image_folder = "/120040051/merged0728"
    append = " Please insist your common knowledge of the world for the answer."
    hf_logging.set_verbosity_error()
    eval_model(model_dir=model_dir, model_type=model_type, text_input_path=text_input, image_folder=image_folder, text_output_path=text_output, postfix=append)