import requests
from urllib.parse import urlencode
import urllib
import base64
import json
import time
import os


def generate_image_from_sentence(end_point, authen, sentence, id, image_folder_path, iter=1):
    out_path = image_folder_path + f'/q{id}_{iter}.jpg'
    if os.path.exists(out_path):
        print(f"Image {out_path} already exists, the request is cancelled.")
        return

    # Constructing the payload as a dictionary
    payload = {
        'prompt': sentence,
        'model': 'dall-e-3',
        'n': 1,
        'quality': 'standard',
        'response_format': 'b64_json',
        'size': '1024x1024',
        'style': 'vivid'
    }
    # Converting the payload dictionary into a URL-encoded string
    encoded_payload = urlencode(payload, quote_via=urllib.parse.quote)
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f"Bearer {authen}"
    }

    try:
        response = requests.request("POST", end_point, headers=headers, data=encoded_payload)
        # Ensure that we have a successful response status before proceeding
        response.raise_for_status()
        response_json = response.json()
        image_base64 = response_json["data"][0]["b64_json"] 
        image_data = base64.b64decode(image_base64)
        with open(out_path, 'wb') as f:
            f.write(image_data)

    except requests.exceptions.RequestException as e:
        # Handle specific request errors (e.g., connectivity, timeout errors)
        print(f"Request failed: {e}")
    except Exception as e:
        # Handle other possible exceptions
        print(f"An error occurred: {e}")


def read_sentence_list(file_path, sent_key_name):
    sent_list = []
    with open(file_path, 'r') as f:
        q_data = json.load(f)
    for q_dict in q_data:
        sent_list.append(q_dict[sent_key_name])
    return sent_list


def write_meta_file(image_folder_path, sent_list):
    out_file_path = image_folder_path + '/meta_info.json'
    if os.path.exists(out_file_path):
        print(f"Meta file {out_file_path} already exists.")
        return
    out_dict = {}
    for index in range(len(sent_list)):
        out_dict[f"image{index+1}"] = sent_list[index]
    with open(out_file_path, 'w') as f:
        json.dump(out_dict, f, indent=4)
    print(f"Generated meta file {out_file_path}")


def deprecate_flawed_image(image_folder, deprecated_list, postfix='.jpg'):
    for item in deprecated_list:
        img_path = image_folder + f'/q{item[0]}_{item[1]}{postfix}'
        dep_img_path = image_folder + f'/q{item[0]}_{item[1]}_dp{postfix}'
        if os.path.exists(img_path):
            ctr = 1
            while os.path.exists(dep_img_path):
                dep_img_path = image_folder + f'/q{item[0]}_{item[1]}_dp{ctr}{postfix}'
                ctr +=1
                if ctr >= 100:
                    print(f"Trying to rename to file {dep_img_path}")
                    raise ValueError("Trying too many times, something must be wrong. Program aborting.")
            os.rename(img_path, dep_img_path)
        else:
            print(f"{img_path} does not exist.")
            continue



if __name__ == '__main__':
    url = "https://cn2us02.opapi.win/v1/images/generations"
    input_file = "/home/liu/test_resources/input_questions/verb_questions_0329.json"
    auth_key = '' # add here
    output_image_folder = '/home/liu/test_resources/images/verb'
    text_list = read_sentence_list(input_file, 'text')
    write_meta_file(output_image_folder, text_list)

    # first two round of generation
    for i, text in enumerate(text_list):
        for it in range(2):
            generate_image_from_sentence(url, auth_key, text, i+1, output_image_folder, iter=it+1) 
            print(f"Generated the {i+1}th image at iteration {it+1}")
            time.sleep(10)
    
    # Makeup rounds
    # flaw_list = [(5, 1), (6, 2), (8, 1), (10, 2), (16, 1), (16, 2)]  # manual check and input (flawed_image_id, iter) to the list
    # deprecate_flawed_image(output_image_folder, flaw_list)
    # for item in flaw_list:
    #     index = item[0]-1
    #     generate_image_from_sentence(url, auth_key, text_list[index], item[0], output_image_folder, item[1])
    #     print(f"Generated the {item[0]}th image.")
    #     time.sleep(10)