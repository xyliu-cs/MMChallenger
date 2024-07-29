from eval import read_json, write_json
from templates import OPENAI_CHECK_ANS_PREFIX_ACTION
from tqdm import tqdm
import json, os, time, requests, datetime

def check_mcq_correctness(model_ans, label_ans):
    sanity_list = [label_ans, f'({label_ans})']
    for ans in sanity_list:
        if model_ans.startswith(ans):
            return True
    return False

def check_yn_correctness(model_ans, label_ans):
    return label_ans.lower() in model_ans.lower()

# Result is not satisfying !!!
# def check_sa_correctness_auto(basic_str, prefix):
#     text_prompt = prefix + basic_str
#     # print(basic_str)
#     url = "https://aigptx.top/v1/chat/completions"
#     api_key = os.environ.get('MY_API_KEY')
#     payload = json.dumps({
#         "model": "gpt-4o-mini",
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant."
#             },
#             {
#                 "role": "user",
#                 "content": text_prompt
#             }
#         ]
#     })
#     headers = {
#         'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
#         'Content-Type': 'application/json',
#         "Authorization": f"Bearer {api_key}"
#     }

#     attempt = 0
#     while attempt < 3:
#         try:
#             response = requests.request("POST", url, headers=headers, data=payload)
#             response.raise_for_status()  # Raises exception for HTTP errors
#             res_dict = response.json()
#             result = res_dict['choices'][0]['message']['content']
#             # print(result)
#             break
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed: {str(e)}")
#             time.sleep(5)  # Wait before retrying
#             attempt += 1
#     if attempt >= 3:
#         print("Maximum retry attempts reached, giving up.")
#         return "Error"
        
#     time.sleep(2)
#     return result

def check_sa_correctness_human(basic_str):
    print(basic_str)
    eval_result = input("Enter 0 for incorrect, 1 for partially correct, 2 for correct. Separate with white space.\n")
    while eval_result not in ['0', '1', '2']:
        eval_result = input("Invalid input. Please enter 0 for incorrect, 1 for partially correct, 2 for correct. Separate with white space.\n")
    return eval_result


def find_mcq_errors(output_list: list, target="all") -> list:
    assert target in ["all", "action", "place"]
    ret_list = []
    if target == "all":
        for output_dict in output_list:
            id = output_dict["id"]
            category = output_dict["category"]
            mcq_ans = output_dict["MCQ_ans"]
            model_ans_lists = output_dict["mcq_model_ans"]
            local_dict = {"id": id, "category": category, "image": []}
            for iid, ans_list in enumerate(model_ans_lists):
                for ans in ans_list:
                    if not check_mcq_correctness(ans, mcq_ans):
                        local_dict["image"].append(iid+1)
                        break
            if local_dict["image"]:
                ret_list.append(local_dict)
    else:
        for output_dict in output_list:
            if output_dict["category"] == target:
                id = output_dict["id"]
                category = output_dict["category"]
                mcq_ans = output_dict["MCQ_ans"]
                model_ans_lists = output_dict["mcq_model_ans"]
                local_dict = {"id": id, "category": category, "image": []}
                for iid, ans_list in enumerate(model_ans_lists):
                    for ans in ans_list:
                        if not check_mcq_correctness(ans, mcq_ans):
                            local_dict["image"].append(iid+1)
                            break
                if local_dict["image"]:
                    ret_list.append(local_dict)
    return ret_list


def find_yn_errors(output_list: list, target="all") -> list:
    assert target in ["all", "action", "place"]
    ret_list = []
    if target == "all":
        for output_dict in output_list:
            id = output_dict["id"]
            category = output_dict["category"]
            model_ans_lists = output_dict["yn_model_ans"]
            local_dict = {"id": id, "category": category, "image": []}
            for iid, ans_list in enumerate(model_ans_lists):
                for ans in ans_list:
                    if not check_yn_correctness(ans, "Yes"):
                        local_dict["image"].append(iid+1)
                        break
            if local_dict["image"]:
                ret_list.append(local_dict)
    else:
        for output_dict in output_list:
            if output_dict["category"] == target:
                id = output_dict["id"]
                category = output_dict["category"]
                model_ans_lists = output_dict["yn_model_ans"]
                local_dict = {"id": id, "category": category, "image": []}
                for iid, ans_list in enumerate(model_ans_lists):
                    for ans in ans_list:
                        if not check_yn_correctness(ans, "Yes"):
                            local_dict["image"].append(iid+1)
                            break
                if local_dict["image"]:
                    ret_list.append(local_dict)
    return ret_list


# def standardize_status(input_string):
#     keywords = ["Correct", "Acceptable", "Incorrect"]
#     for keyword in keywords:
#         if keyword in input_string:
#             return keyword


def find_sa_errors(output_list: list, target="all", eval_type="human") -> list:
    assert target in ["all", "action", "place"]
    ret_list = []
    if target == "all":
        eval_output_list = output_list
    elif target == "action":
        eval_output_list = [output_dict for output_dict in output_list if output_dict["category"] == "action"]
    elif target == "place":
        eval_output_list = [output_dict for output_dict in output_list if output_dict["category"] == "place"]

    if eval_type == "human":
        evaler = input("Please enter your name in english: \n")
        eval_email = input("Please enter your email address: \n")
        start_time = datetime.datetime.now()

    for idx, output_dict in enumerate(eval_output_list[:]):
        total = len(eval_output_list)
        id = output_dict["id"]
        category = output_dict["category"]
        model_ans_lists = output_dict["sa_model_ans"]
        local_dict = {"id": id, "category": category, "image": [], "info": '',"human_eval": []}
        ans_str_list = []
        for iid, ans_list in enumerate(model_ans_lists):
            ans_set = list(set(ans_list))
            ans_string = f"[Model answer {iid+1}] {', '.join(ans_set)}"
            ans_str_list.append(ans_string)
        total_ans_str = '\n'.join(ans_str_list)
        # rebuild clean question string
        mcq_str = output_dict["mcq"]
        ques_str = "[Question] " + mcq_str.split("Question:")[1].split("Options:")[0].strip()
        ground_truth = "[Ground truth] " + list(output_dict["target"].values())[0]
        basics = f"{ques_str}\n{ground_truth}\n{total_ans_str}"
        result = check_sa_correctness_human(basics)
        results = result.split(' ')
        for i, result_digit in enumerate(results):
            local_dict["image"].append(i+1)
            local_dict["info"] = basics
            local_dict["human_eval"].append(result_digit)
        ret_list.append(local_dict)
        print(f"Progress: {idx+1}/{total} labeled.")
        print("\n")

    end_time = datetime.datetime.now()
    ret_list.append({"evaluator": evaler, "evaluator_email": eval_email, "start_time": str(start_time), "end_time": str(end_time), "time_used": str(end_time - start_time)})
    return ret_list


if __name__ == "__main__":
    # output_path = "/120040051/test_resource/output_answers/llava15_7b_outputs_fmt.json"
    # output_path = "/120040051/test_resource/output_answers/instructblip_7b_outputs.json"
    output_path = "/120040051/test_resource/output_answers/qwen_vl_chat_outputs.json"

    outputs = read_json(output_path)
    mcq_error_list = find_mcq_errors(outputs)
    yn_error_list = find_yn_errors(outputs)

    # print(mcq_error_list)
    print("mcq all: ", len(mcq_error_list))
    print("mcq all pct: ", round(len(mcq_error_list)/344, 4))
    mcq_error_list_action = find_mcq_errors(outputs, target="action")
    count = 0
    for info_dict in mcq_error_list:
        for output_dict in outputs:
            if output_dict['id'] == info_dict['id'] and output_dict['category'] == info_dict['category']:
                for model_ans_list in output_dict["mcq_model_ans"]:
                    for ans in model_ans_list:
                        found = False
                        if ans == "":
                            count += 1
                            found = True
                            print(f"id = {output_dict['id']}, category = {output_dict['category']}")
                            break
                    if found:
                        break
    print("Empty ans:", count)
                    
    print("mcq action: ", len(mcq_error_list_action))
    print("mcq action pct: ", round(len(mcq_error_list_action)/156, 4))
    mcq_error_list_place = find_mcq_errors(outputs, target="place")
    print("mcq place: ", len(mcq_error_list_place))
    print("mcq place pct: ", round(len(mcq_error_list_place)/188, 4))

    # print(yn_error_list)
    print("yn all: ", len(yn_error_list))
    print("yn all pct: ", round(len(yn_error_list)/344, 4))
    yn_error_list_action = find_yn_errors(outputs, target="action")
    print("yn action: ", len(yn_error_list_action))
    print("yn action pct: ", round(len(yn_error_list_action)/156, 4))
    yn_error_list_place = find_yn_errors(outputs, target="place")
    print("yn place: ", len(yn_error_list_place))
    print("yn place pct: ", round(len(yn_error_list_place)/188, 4))

    # print('\n')
    # print(mcq_error_list)

    # sa_eval_list = find_sa_errors(output_list=outputs, target="all")
    # write_json("llava15_13b_sa_human_eval.json", sa_eval_list)

    # eval_res = read_json("llava15_13b_sa_human_eval.json")
    # total = 0
    # action = 0
    # place = 0
    # assert len(eval_res) == 345
    # for eval_dict in eval_res[:-1]:
    #     if eval_dict["human_eval"][0] == "0":
    #         total += 1
    #         if eval_dict["category"] == "action":
    #             action += 1
    #         elif eval_dict["category"] == "place":
    #             place += 1
    # print("sa all: ", total)
    # print("sa action: ", action)
    # print("sa place: ", place)