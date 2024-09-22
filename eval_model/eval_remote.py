import time, base64, json, requests, os, copy
from tqdm import tqdm
from PIL import Image
from templates import route_templates
from prompt import generate_mcq_prompt, generate_yn_prompt, generate_sa_prompt
from prompt import generate_mcq_prompt_st, generate_yn_prompt_st, generate_sa_prompt_st
from utils import read_json, write_json, infer_target_type, base64_encode_img


# use openai input format for all models as required by the third party api provider
def build_query(model_name, image_path, text_prompt, max_tokens=100, style='gpt', use_img=True):
    api_key = os.environ.get('API_KEY')   # use export API_KEY = 'your key'
    if style == 'gpt':
        if use_img:
            b64_img = base64_encode_img(image_path=image_path)
            img_dict = { 
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{b64_img}"      # remember to change the image types accordingly
                            }   
                        }
            prompt_dict = {"type": "text", "text": text_prompt}
            user_content = [img_dict, prompt_dict]
        else:
            prompt_dict = {"type": "text", "text": text_prompt}
            user_content = [prompt_dict]

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
            "max_tokens": max_tokens,
            "stream": False
        }
    elif style == 'anthropic':
        if use_img: 
            b64_img = base64_encode_img(image_path=image_path, resize=True)    # resize to 800 x 800
            image_dict = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_img
                            }
                        }
            text_dict = {
                        "type": "text",
                        "text": text_prompt
                    }
            user_content = [image_dict, text_dict]
        else:
            text_dict = {
                        "type": "text",
                        "text": text_prompt
                    }
            user_content = [text_dict]
           
        headers = {
            "User-Agent": 'Apifox/1.0.0 (https://apifox.com)',
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",  # anthropic api version
            "Authorization": f"Bearer {api_key}"
        }
        messages = [
            {
                "role": "user",
                "content": user_content
            }
        ]
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens
        }

    query = {"headers": headers, "payload": payload}
    # print(query)
    return query


def post_request(input_dict, end_point, max_retry=5, timeout=30):
    attempt = 0
    while attempt < max_retry:
        try:
            response = requests.request("POST", url=end_point, headers=input_dict["headers"], data=json.dumps(input_dict["payload"]), timeout=timeout)
            response.raise_for_status()  # Raises exception for HTTP errors
            return response
        except requests.exceptions.Timeout:
            print(f"Attempt {attempt + 1} failed: Request timed out.")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
        time.sleep(5)  # Wait before retrying
        attempt += 1
    print("Maximum retry attempts reached, returning an empty string.")
    return ""


def parse_response(response, model_type):
    if model_type == 'gpt':
        try:
            res_json = response.json()
            # openai response format
            output_text = res_json['choices'][0]['message']['content'].strip() 
            return output_text
        except Exception as e:
            print(f"Parsing response failed: {str(e)}\nReturn the raw response instead.")
            output_text = response.text.strip()
            return output_text
    elif model_type == "anthropic":
        try:
            res_json = response.json()
            # anthropic response format
            output_text = res_json['content'][0]['text'].strip()
            return output_text
        except Exception as e:
            print(f"Parsing response failed: {str(e)}\nReturn the raw response instead.")
            output_text = response.text.strip()
            return output_text


def eval_model_api(model_name, model_type, end_url, text_input_path, image_folder, 
                   text_output_path, prefix='', postfix='', repeat=1, sanity_test=False, 
                   use_cot=False, max_tokens=10, use_img=True):
    input_list_of_dict = read_json(text_input_path)
    output_list_of_dict = []
    with open(text_output_path, "w") as f:
        f.write("[\n")
        for idx, input_dict in tqdm(enumerate(input_list_of_dict[:]), total=len(input_list_of_dict), desc="Infering answers"):
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
            model_templates = route_templates(model_type=model_type, use_cot=use_cot, sanity_test=sanity_test)
            if sanity_test:
                mcq = generate_mcq_prompt_st(gt_triplet=gt_triplet, mcq_dict=input_dict["MCQ_options"], target=input_type, 
                                      mcq_prompt_template=model_templates["MCQ"], postfix=postfix)
                yn = generate_yn_prompt_st(gt_triplet=gt_triplet, yn_prompt_template=model_templates["YN"], target=input_type, 
                                        prefix=prefix, postfix=postfix)
                sa = generate_sa_prompt_st(gt_triplet=gt_triplet, sa_prompt_template=model_templates["SA"], target=input_type, 
                                        prefix=prefix, postfix=postfix)
            else:
                mcq = generate_mcq_prompt(gt_triplet=gt_triplet, mcq_dict=input_dict["MCQ_options"], target=input_type, 
                                        mcq_prompt_template=model_templates["MCQ"], postfix=postfix)
                yn = generate_yn_prompt(gt_triplet=gt_triplet, yn_prompt_template=model_templates["YN"], target=input_type, 
                                        prefix=prefix, postfix=postfix)
                sa = generate_sa_prompt(gt_triplet=gt_triplet, sa_prompt_template=model_templates["SA"], target=input_type, 
                                        prefix=prefix, postfix=postfix)
            prompt_dict = {"mcq": mcq, "yn": yn, "sa": sa}
            output_dict = copy.deepcopy(input_dict)
            output_dict["category"] = input_type
            for q_type, prompt in prompt_dict.items():
                output_dict[q_type] = prompt
                output_dict[f"{q_type}_model_ans"] = []
                for i, image_path in enumerate(image_paths):
                    local_ans = []
                    for iter in range(repeat):
                        query = build_query(model_name=model_name, image_path=image_path, text_prompt=prompt, 
                                            style=model_type, max_tokens=max_tokens, use_img=use_img)
                        print("Query: ", query['payload']['messages'][0]['content'][0]['text'])
                        response = post_request(input_dict=query, end_point=end_url)
                        output_text = parse_response(response=response, model_type=model_type)
                        print(output_text)
                        time.sleep(1)
                        local_ans.append(output_text)
                    output_dict[f"{q_type}_model_ans"].append(local_ans)
            f.write(json.dumps(output_dict, indent=4))
            if idx != len(input_list_of_dict) - 1:
                f.write(",\n")
            f.flush()
            output_list_of_dict.append(output_dict)
        f.write('\n]')
        f.flush()   
        print(f"Finished on all the {len(output_list_of_dict)} inputs")
    return output_list_of_dict



if __name__ == "__main__":
    model_name = "claude-3-5-sonnet-20240620"
    # model_name = "claude-3-sonnet"
    # model_name = "gpt-4o-2024-05-13"


    # model_type = "gpt"
    model_type = "anthropic"
    
    # end_url = "https://aigptx.top/v1/chat/completions"
    end_url_2 = "https://cn2us02.opapi.win/v1/messages"

    text_input_path = "/120040051/test_resource/merged0728/input_info.json"
    image_folder = "/120040051/test_resource/merged0728/"
    # out_path = "/home/liu/test_resources/output_answers/GPT-4o/gpt-4o-2024-05-13_outputs_10.json"
    # out_path = "/120040051/Github_Repos/VKConflict/eval_model/sanity_test_results/gpt-4o-2024-05-13_sanity_test.json"
    out_path = f"/120040051/Github_Repos/VKConflict/eval_model/sanity_test_results/{model_name}_sanity_test.json"
    
    # postfix = "Please insist your common knowledge of the world. "
    eval_model_api(model_name=model_name, model_type=model_type, end_url=end_url_2, 
                   text_input_path=text_input_path, image_folder=image_folder, 
                   text_output_path=out_path, use_cot=False, repeat=1, max_tokens=20, 
                   sanity_test=True, use_img=False)