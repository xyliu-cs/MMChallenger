from utils import read_json, write_json
from templates import OPENAI_CHECK_ANS_PREFIX_ACTION
from tqdm import tqdm
import json, os, time, requests, datetime, shutil

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


def eval_sa_answers(output_list: list, target="all", eval_type="human") -> list:
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


def find_sa_errors(evaled_ans_list: list) -> list:
    # human evaled, last dict is meta info
    if "evaluator" in evaled_ans_list[-1]:
        ret_list = []
        for eval_dict in evaled_ans_list[:-1]:
            if eval_dict["human_eval"][0] == "0":
                ret_list.append({"id": eval_dict["id"], "category": eval_dict["category"], "image": eval_dict["image"]})
        return ret_list


def build_path_list(model_names: list) -> tuple:
    obj_results = []
    sub_evaled_results = []
    for model_name in model_names:
        obj_results.append(model_name + '_outputs.json')
        sub_evaled_results.append(model_name + '_sa_human_eval.json')
    return obj_results, sub_evaled_results    


def write_err_stats(obj_res_list, sub_res_list, res_folder, output_file='error_stats.json'):
    assert len(obj_res_list) == len(sub_res_list), f"Unequal length of input lists {len(obj_res_list)} and {len(sub_res_list)}"
    info_dict, global_yn, global_mc, global_sa, global_all = {}, {}, {}, {}, {}
    for obj_path, sub_path in list(zip(obj_res_list, sub_res_list)):
        model_name = obj_path.split('_outputs')[0]
        obj_full_path = os.path.join(res_folder, obj_path)
        sub_full_path = os.path.join(res_folder, sub_path)
        
        assert os.path.isfile(obj_full_path), f"Invalid file path {obj_full_path}"
        assert os.path.isfile(sub_full_path), f"Invalid file path {sub_full_path}"
        
        objective_qa = read_json(obj_full_path)
        subjective_qa = read_json(sub_full_path)

        mc_error_list = find_mcq_errors(objective_qa)
        yn_error_list = find_yn_errors(objective_qa)
        sa_error_list = find_sa_errors(subjective_qa)

        # should serve as the unique key, 1 is the count for formatting purpose
        pure_id_list_yn = [ [item["category"][0].upper() + str(item["id"]), 1] for item in yn_error_list ]
        global_yn = add_and_increment(pure_id_list_yn, global_yn)
        pure_id_list_mc = [ [item["category"][0].upper() + str(item["id"]), 1] for item in mc_error_list ]
        global_mc = add_and_increment(pure_id_list_mc, global_mc)
        pure_id_list_sa = [ [item["category"][0].upper() + str(item["id"]), 1] for item in sa_error_list ]
        global_sa = add_and_increment(pure_id_list_sa, global_sa)
        
        pure_id_list_all = pure_id_list_yn + pure_id_list_mc + pure_id_list_sa
        global_all = add_and_increment(pure_id_list_all, global_all)

        info_dict[model_name] = {'yn_error': pure_id_list_yn, 'mc_error': pure_id_list_mc, 'sa_error': pure_id_list_sa}

    global_yn = sorted(global_yn.items(), key=lambda x: x[1], reverse=True)
    global_mc = sorted(global_mc.items(), key=lambda x: x[1], reverse=True)
    global_sa = sorted(global_sa.items(), key=lambda x: x[1], reverse=True)
    global_all = sorted(global_all.items(), key=lambda x: x[1], reverse=True)
    info_dict["global"] = {'yn_error': global_yn, 'mc_error': global_mc, 'sa_error': global_sa, 'all_error': global_all}
    
    output_path = os.path.join(res_folder, output_file)
    with open(output_path, 'w') as f:
        json.dump(info_dict, f)    
    print(f"Successfully write error statistics to {output_path}.") 


def add_and_increment(add_list: list, base_dict: dict) -> dict:
    for item in add_list:
        unikey = item[0]
        if unikey in base_dict:
            base_dict[unikey] += 1
        else:
            base_dict[unikey] = 1
    return base_dict


def ret_input_info_upon_cond(lookup_list: list, cond_list: list):
    revert_match = {'A': 'action', 'P': 'place'}
    ret_info_list = []
    for item in cond_list:
        unique_id, freq = item[0], item[1]
        cat = revert_match[unique_id[0]]
        id = int(unique_id[1:])
        found = False
        for info_dict in lookup_list:
            if info_dict["id"] == id and list(info_dict["target"].keys())[0] == cat:
                ret_info_list.append(info_dict)
                found = True
                break
        assert found, f"Item of unique id {unique_id} should be found in the input info list"
    return ret_info_list


def copy_imgs(info_list: list, source_folder: str, target_folder: str) -> None:
    for info_dict in info_list:
        images = info_dict["image"]
        for image_name in images:
            old_path = os.path.join(source_folder, image_name)
            assert os.path.isfile(old_path), f"{old_path} does not exist!"
            new_path = os.path.join(target_folder, image_name)
            if not os.path.exists(new_path):
                shutil.copy(old_path, new_path)
                print(f"Copied image to {new_path}") 
            else:
                print(f"{new_path} already exists")




if __name__ == "__main__":

    # | ------------------------------------- |
    # |  PART 1: Eval and Collect Errors      |
    # | ------------------------------------- |
    # output_path = "/120040051/test_resource/output_answers/llava15_7b_outputs_fmt.json"
    # output_path = "/120040051/test_resource/output_answers/instructblip_7b_outputs.json"
    # output_path = "llama3-llava-next-8b_outputs.json"

    # outputs = read_json(output_path)

    # mcq_error_list = find_mcq_errors(outputs)
    # yn_error_list = find_yn_errors(outputs)
    # print(yn_error_list)
    # print("yn all: ", len(yn_error_list))
    # print("yn all pct: ", round(1 - len(yn_error_list)/344, 4))
    # yn_error_list_action = find_yn_errors(outputs, target="action")
    # print("yn action: ", len(yn_error_list_action))
    # print("yn action pct: ", round(1 - len(yn_error_list_action)/156, 4))
    # yn_error_list_place = find_yn_errors(outputs, target="place")
    # print("yn place: ", len(yn_error_list_place))
    # print("yn place pct: ", round(1 - len(yn_error_list_place)/188, 4))

    # # print(mcq_error_list)
    # print("mcq all: ", len(mcq_error_list))
    # print("mcq all pct: ", round(1 - len(mcq_error_list)/344, 4))
    # mcq_error_list_action = find_mcq_errors(outputs, target="action")
    # count = 0
    # for info_dict in mcq_error_list:
    #     for output_dict in outputs:
    #         if output_dict['id'] == info_dict['id'] and output_dict['category'] == info_dict['category']:
    #             for model_ans_list in output_dict["mcq_model_ans"]:
    #                 for ans in model_ans_list:
    #                     found = False
    #                     if ans == "":
    #                         count += 1
    #                         found = True
    #                         print(f"id = {output_dict['id']}, category = {output_dict['category']}")
    #                         break
    #                 if found:
    #                     break
    # print("Empty ans:", count)

    # print("mcq action: ", len(mcq_error_list_action))
    # print("mcq action pct: ", round(1 - len(mcq_error_list_action)/156, 4))
    # mcq_error_list_place = find_mcq_errors(outputs, target="place")
    # print("mcq place: ", len(mcq_error_list_place))
    # print("mcq place pct: ", round(1 - len(mcq_error_list_place)/188, 4))

    # human eval
    # sa_eval_list = eval_sa_answers(output_list=outputs, target="all")
    # write_json("llava-v1.6-34b_outputs_human_eval.json", sa_eval_list)

    # eval_res = read_json("llama3-llava-next-8b_outputs_human_eval.json")
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
    # print("sa all (%): ", round(1 - total/344, 4))

    # print("sa action: ", action)
    # print("sa all (%): ", round(1 - action/156, 4))

    # print("sa place: ", place)
    # print("sa all (%): ", round(1 - place/188, 4))


    # | ------------------------------------- |
    # |  PART 2: Collect Error Statistics     |
    # | ------------------------------------- |
    model_names = ["llama3-llava-next-8b", "llava-v1.6-34b", "llava-1.5-13b", 
                   "blip2_t5_xxl", "instructblip-flan-t5-xxl", "qwen-vl",
                   "qwen-vl-chat", "gpt-4o", "claude-3-5-sonnet"]
    
    objective_paths, subjective_paths = build_path_list(model_names)
    write_err_stats(obj_res_list=objective_paths, sub_res_list=subjective_paths, res_folder='results')

    # | ------------------------------------- |
    # |  PART 3: Construct Challenge Set      |
    # | ------------------------------------- |
    err_stats_file = './results/error_stats.json'
    original_input_folder_path_full =  r"C:\Users\xiaoyuanliu\Desktop\merged0728"
    check_model = "llama3-llava-next-8b"

    lookup_file = os.path.join(original_input_folder_path_full, 'input_info.json')
    # model name or global
    check_model = "llama3-llava-next-8b"
    challenge_set = 'challset_' + check_model 
    challset_path = os.path.join(os.path.dirname(original_input_folder_path_full), challenge_set)
    # overwrite
    if os.path.exists(challset_path):
        shutil.rmtree(challset_path)
    os.makedirs(challset_path)
    print(f"Directory {challset_path} created")
    
    err_stats = read_json(err_stats_file)
    lookup_info = read_json(lookup_file)
    
    target_dict = err_stats[check_model]
    target_err_yn = target_dict["yn_error"]
    target_err_mc = target_dict["mc_error"]
    target_err_sa = target_dict["sa_error"]
    # target_err_all = target_dict["all_error"]
    
    threshold = 1
    filtered_errs = [item for item in target_err_sa if item[1] >= threshold]

    challset_info = ret_input_info_upon_cond(lookup_list=lookup_info, cond_list=filtered_errs)
    copy_imgs(challset_info, original_input_folder_path_full, challset_path)
    out_info_path = os.path.join(challset_path, 'input_info.json')
    write_json(out_info_path, challset_info)

    print(f"error instances   (count >= {threshold}): {len(target_err_sa)}")
    print(f"error instances % (count >= {threshold}): {len(target_err_sa)/344}")
    print(f"error instances list: \n{target_err_sa}")