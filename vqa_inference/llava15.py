import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import json
import os

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("/120040051/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained("/120040051/llava-1.5-7b-hf", device_map = 'auto')

input_file_path = "/120040051/test_resource/input_question/0502/location_questions_NT_0502.json"
output_file_path = "/120040051/test_resource/output_answers/location_answers_NT_0502_llava15-7b.json"
image_folder = "/120040051/test_resource/images/0420/loc_webp"

with open(input_file_path, 'r') as f:
    data = json.load(f)

with open(output_file_path, 'w') as f:
    json.dump({'a': 10}, f, indent=4) # test availability

answers = []
for question in data:
    local_answer = question.copy()
    q_id = question['id']
    print(f"[Asking Question {q_id}] ...")

    target_keys = ['choice llava', "binary-yes", "binary-no", "binary-cp", "open llava"]

    image_1_path = f"{image_folder}/q{q_id}_1.webp"
    image_2_path = f"{image_folder}/q{q_id}_2.webp"

    if os.path.isfile(image_1_path) and os.path.isfile(image_2_path):
        image_list = [image_1_path, image_2_path]
        for img_id in range(2):
            raw_image = Image.open(image_list[img_id])
            for q_key in target_keys:
                q_string = question[q_key]
                q_prompt = f"USER: <image>\n{q_string}\nASSISTANT:"
                for iter in range(5): # test over 5 trials over each question
                    inputs = processor(q_prompt, raw_image, return_tensors='pt').to(device)
                    output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.2)[0]
                    output = processor.decode(output_ids, skip_special_tokens=True)
                    answer = output.split('ASSISTANT: ')[1]
                    if q_key == 'choice llava':
                        answer_key = f"choice_ans_img{img_id+1}_{iter+1}"
                    elif q_key == 'open llava':
                        answer_key = f"open_ans_img{img_id+1}_{iter+1}"
                    else:
                        answer_key = f"{q_key}_ans_img{img_id+1}_{iter+1}"
                    
                    local_answer[answer_key] = answer

        answers.append(local_answer)   
    
    else:
        print(f"Image path {image_1_path} or {image_2_path} does not exist!")


with open(output_file_path, 'w') as f:
    json.dump(answers, f, indent=4)