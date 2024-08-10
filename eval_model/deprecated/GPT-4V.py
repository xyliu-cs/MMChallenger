import time
import base64
import json
import requests


mirror_url = "https://cn2us02.opapi.win/v1/chat/completions"
my_key = "sk-HcTUQUYI386890171282T3BlBKFJ876AE6081bd74D0c8871"

input_file = '/home/liu/test_resources/input_questions/location_questions_0716_base.json'
input_image_folder = '/home/liu/test_resources/images/0716/loc_10'
output_file = '/home/liu/test_resources/output_answers/GPT-4V/location_answers_0716_VP_gpt4v.json'

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
with open(input_file, 'r') as f:
    data = json.load(f)

output_dict = []
for question_dict in data[:10]:
    local_ans = question_dict.copy()
    print(f"Processing question {question_dict['id']}")

    for img_id in range(2):
        image_path = f"{input_image_folder}/q{question_dict['id']}_{img_id+1}.webp"
        for question_key in ['choice llava', 'binary-yes', 'binary-no', 'binary-cp', 'open llava']:
            question = question_dict[question_key]
            b64_img = encode_image(image_path)
            img_dict = { "type": "image_url",
                        "image_url": {
                            "url": f"data:image/webp;base64,{b64_img}"   # remember to change the image types accordingly
                            }
                        }
            prompt_dict = {"type": "text","text": question}
            user_content = [img_dict, prompt_dict]
            
            headers = {
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                "Content-Type": "application/json",
                "Authorization": f"Bearer {my_key}"
            }

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                "max_tokens": 100
            }

            attempt = 0
            while attempt < 3:
                try:
                    response = requests.post(mirror_url, headers=headers, json=payload)
                    response.raise_for_status()  # Raises exception for HTTP errors
                    res_dict = response.json()
                    answer = res_dict['choices'][0]['message']['content']
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(5)  # Wait before retrying
                    attempt += 1
            else:
                print("Maximum retry attempts reached, moving to next.")
                continue

            answer = res_dict['choices'][0]['message']['content']
            if question_key == 'choice llava':
                answer_key = f'choice_ans_img{img_id+1}_1'
            elif question_key == 'open llava':
                answer_key = f'open_ans_img{img_id+1}_1'
            else:
                answer_key = f'{question_key}_ans_img{img_id+1}_1'
            local_ans[answer_key] = answer
            time.sleep(5)
    
    output_dict.append(local_ans)

with open(output_file, 'w') as f:
    json.dump(output_dict, f, indent=4)