from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, AutoModelForVision2Seq, InstructBlipConfig
from accelerate import init_empty_weights, infer_auto_device_map
import torch
from PIL import Image
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model configuration.
config = InstructBlipConfig.from_pretrained("/120040051/instructblip-vicuna-7b")
with init_empty_weights():
    model = AutoModelForVision2Seq.from_config(config)
    model.tie_weights()

# Infer device map based on the available resources.
device_map = infer_auto_device_map(model, max_memory={0: "25GiB", 1: "25GiB"}, no_split_module_classes=['InstructBlipEncoderLayer', 'InstructBlipQFormerLayer', 'LlamaDecoderLayer'])
device_map['language_model.lm_head'] = device_map['language_projection'] = device_map[('language_model.model.embed_tokens')]
# device_map['llm_model.model.embed_tokens'] = device_map['llm_model.lm_head'] = device_map['llm_proj']

# device_map
# offload = ""
# # Load the processor and model for image processing.
processor = InstructBlipProcessor.from_pretrained("/120040051/instructblip-vicuna-7b", device_map="auto")
model = InstructBlipForConditionalGeneration.from_pretrained("/120040051/instructblip-vicuna-7b",
                                                             device_map=device_map)


with open('/120040051/test_resource/input_question/verb_questions_vicuna_0308_mod.json', 'r') as f:
    data = json.load(f)

with open("/120040051/test_resource/output_answers/verb_answers_vicuna_0308_IB.json", 'w') as f:
    json.dump({'a': 10}, f, indent=4) # test availability

verb_answers = []
for question in data[:]:
    local_answer = question.copy()
    q_id = question['id']
    print(f"[Asking Question {q_id}] ...")

    choice_question_1 = question['choice']
    choice_question_2 = question['choice2']

    binary_true = question["binary-yes"]
    binary_false = question["binary-no"]
    binary_compare = question["binary-cp"]

    open_question = question["open"]

    image_1_path = f"/120040051/test_resource/images/verb_0308/q{q_id}_1.webp"
    image_2_path = f"/120040051/test_resource/images/verb_0308/q{q_id}_2.webp"

    if os.path.isfile(image_1_path) and os.path.isfile(image_2_path):
        # choice_prompt_1 = f"USER: <image>\n{choice_question_1}\nASSISTANT:"
        # choice_prompt_2 = f"USER: <image>\n{choice_question_2}\nASSISTANT:"
        # binary_y_prompt = f"USER: <image>\n{binary_true}\nASSISTANT:"
        # binary_n_prompt = f"USER: <image>\n{binary_false}\nASSISTANT:"
        # binary_c_prompt = f"USER: <image>\n{binary_compare}\nASSISTANT:"
        # open_prompt = f"USER: <image>\n{open_question}\nASSISTANT:"
        image_list = [image_1_path, image_2_path]
        for img_id in range(2):
            raw_image = Image.open(image_list[img_id]).convert("RGB")
            for iter in range(5): # test over 5 trials over each question
                inputs = processor(raw_image, choice_question_1, return_tensors='pt').to(device)
                output_ids = model.generate(**inputs, max_new_tokens=30, min_length=4, do_sample=True, num_beams=3, temperature=0.2, repetition_penalty=1.5, length_penalty=1.0)
                output = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # answer = output.split('ASSISTANT: ')[1]
                local_answer[f"choice1_ans_img{img_id+1}_{iter+1}"] = output

            for iter in range(5): # test over 5 trials over each question
                inputs = processor(raw_image, choice_question_2, return_tensors='pt').to(device)
                output_ids = model.generate(**inputs, max_new_tokens=30, min_length=4, do_sample=True, num_beams=3, temperature=0.2, repetition_penalty=1.5, length_penalty=1.0)
                output = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # answer = output.split('ASSISTANT: ')[1]
                local_answer[f"choice2_ans_img{img_id+1}_{iter+1}"] = output

            for iter in range(5): # test over 5 trials over each question
                inputs = processor(raw_image, binary_true, return_tensors='pt').to(device)
                output_ids = model.generate(**inputs, max_new_tokens=50, min_length=4, do_sample=True, num_beams=3, temperature=0.2, repetition_penalty=1.5, length_penalty=1.0)
                output = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # answer = output.split('ASSISTANT: ')[1]
                local_answer[f"binary-yes_ans_img{img_id+1}_{iter+1}"] = output

            for iter in range(5): # test over 5 trials over each question
                inputs = processor(raw_image, binary_false, return_tensors='pt').to(device)
                output_ids = model.generate(**inputs, max_new_tokens=50, min_length=4, do_sample=True, num_beams=3, temperature=0.2, repetition_penalty=1.5, length_penalty=1.0)
                output = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # answer = output.split('ASSISTANT: ')[1]
                local_answer[f"binary-no_ans_img{img_id+1}_{iter+1}"] = output

            for iter in range(5): # test over 5 trials over each question
                inputs = processor(raw_image, binary_compare, return_tensors='pt').to(device)
                output_ids = model.generate(**inputs, max_new_tokens=50, min_length=4, do_sample=True, num_beams=3, temperature=0.2, repetition_penalty=1.5, length_penalty=1.0)
                output = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # answer = output.split('ASSISTANT: ')[1]
                local_answer[f"binary-cp_ans_img{img_id+1}_{iter+1}"] = output

            for iter in range(5): # test over 5 trials over each question
                inputs = processor(raw_image, open_question, return_tensors='pt').to(device)
                output_ids = model.generate(**inputs, max_new_tokens=128, min_length=4, do_sample=True, num_beams=3, temperature=0.2, repetition_penalty=1.5, length_penalty=1.0)
                output = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # answer = output.split('ASSISTANT: ')[1]
                local_answer[f"open_ans_img{img_id+1}_{iter+1}"] = output
        
        verb_answers.append(local_answer)   
    
    else:
        print(f"Image path {image_1_path} or {image_2_path} does not exist!")


with open("/120040051/test_resource/output_answers/verb_answers_vicuna_0308_IB.json", 'w') as f:
    json.dump(verb_answers, f, indent=4)