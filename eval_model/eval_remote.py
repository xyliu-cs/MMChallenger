import time, base64, json, requests, os, copy
from tqdm import tqdm
from PIL import Image
from templates import route_templates
from prompt import generate_mcq_prompt, generate_yn_prompt, generate_sa_prompt
from utils import read_json, write_json, infer_target_type, base64_encode_img


# use openai input format for all models as required by the third party api provider
def build_query(model_name, image_path, text_prompt, max_tokens=100):
    b64_img = base64_encode_img(image_path=image_path)
    img_dict = { 
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64,{b64_img}"   # remember to change the image types accordingly
                    }   
                }
    prompt_dict = {"type": "text", "text": text_prompt}
    user_content = [img_dict, prompt_dict]
    api_key = os.environ.get('OPENAI_API_KEY')   # use export OPENAI_API_KEY = 'your key'
    headers = {
        "User-Agent": 'Apifox/1.0.0 (https://apifox.com)',
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": user_content
            }
        ],
        "max_tokens": max_tokens
    }
    query = {"headers": headers, "payload": payload}
    return query


def post_request(input_dict, end_point, max_retry=5):
    attempt = 0
    while attempt < max_retry:
        try:
            response = requests.post(end_point, headers=input_dict["headers"], json=input_dict["payload"])
            response.raise_for_status()  # Raises exception for HTTP errors
            res_dict = response.json()
            answer = res_dict['choices'][0]['message']['content'].strip()
            # print(answer)
            return answer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(5)  # Wait before retrying
            attempt += 1
    print("Maximum retry attempts reached, returning an empty string.")
    return ""


def eval_openai_model(model_name, model_type, end_url, text_input_path, image_folder, text_output_path, prefix='', postfix='', repeat=1):
    input_list_of_dict = read_json(text_input_path)
    output_list_of_dict = []
    for input_dict in tqdm(input_list_of_dict[:150], "Infering answers"):
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
                for iter in range(repeat):
                    query = build_query(model_name=model_name, image_path=image_path, text_prompt=prompt)
                    output_text = post_request(input_dict=query, end_point=end_url)
                    time.sleep(1)
                    local_ans.append(output_text)
                output_dict[f"{q_type}_model_ans"].append(local_ans)
        output_list_of_dict.append(output_dict)
    print(f"Finished on all the {len(output_list_of_dict)} inputs")
    write_json(json_list=output_list_of_dict, json_path=text_output_path)



if __name__ == "__main__":
    model_name = "gemini-1.5-pro-001"
    model_type = "gpt"
    gemini_url = "https://aigptx.top/v1/chat/completions"
    text_input_path = "/home/liu/merged0728/input_info.json"
    image_folder = "/home/liu/merged0728"
    out_path = "/home/liu/test_resources/output_answers/gemini-1.5-pro-001/gemini-1.5-pro-001_outputs_150.json"
    eval_openai_model(model_name=model_name, model_type=model_type, end_url=gemini_url, text_input_path=text_input_path, image_folder=image_folder, text_output_path=out_path)