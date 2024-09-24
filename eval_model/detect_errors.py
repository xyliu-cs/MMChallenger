from utils import read_json, write_json
from templates import OPENAI_CHECK_ANS_PREFIX_ACTION
from tqdm import tqdm
import json, os, time, requests, datetime, shutil
import warnings

def find_mcq_errors(output_list: list, target="all", response_prefix='') -> list:
    assert target in ["all", "action", "place"]

    def is_mcq_correct(model_ans, label_ans, label_phrase):
        sanity_list = [label_ans, f'({label_ans})', f'({label_ans}']
        for ans in sanity_list:
            if model_ans.startswith(ans):
                return True
        if (label_phrase in model_ans) and ("not" not in model_ans):
            print(f'[Warning] Correct answer "{label_phrase}" found in model answer "{model_ans}" but not at the beginning.')
            return True
        return False
    
    def is_mcq_direct_incrt(model_ans, label_ans, labels=['A', 'B', 'C', 'D']):
        incorrect_labels = [label for label in labels if label != label_ans]
        for label in incorrect_labels:
            check_list = [label, f'({label}']
            if any([model_ans.startswith(check) for check in check_list]):
                return True
        return False

    def human_eval_mcq(question, model_ans, label_ans, mcq_options):
        print(f"Question: {question}")
        print(f"Label answer: {label_ans} {mcq_options[label_ans]}")
        print(f"Model answer: {model_ans}")
        input_res = input("Enter 0 for incorrect, 1 for correct: ")
        while input_res not in ['0', '1']:
            input_res = input("Invalid input. Please enter 0 for incorrect, 1 for correct: ")
        if input_res == '0':
            return False
        return True

    if target == "all":
        examine_list = output_list
    elif target == "action":
        examine_list = [output_dict for output_dict in output_list if output_dict["category"] == "action"]
    elif target == "place":
        examine_list = [output_dict for output_dict in output_list if output_dict["category"] == "place"]
    
    ret_list = []
    mcq_error_count = 0
    for output_dict in examine_list:
        id = output_dict["id"]
        category = output_dict["category"]
        mcq_ans = output_dict["MCQ_ans"]
        model_ans_lists = output_dict["mcq_model_ans"]
        mcq_options = output_dict["MCQ_options"]
        local_dict = {"id": id, "category": category, "ans_idx": []}
        for iid, ans_list in enumerate(model_ans_lists):
            # ans_list = [ans.split(response_prefix)[1].strip() if response_prefix in ans else ans for ans in ans_list] # for cot only
            for ans in ans_list:
                if response_prefix:
                    if response_prefix in ans:
                        ans = ans.split(response_prefix)[-1].replace(':', '').replace('"', '').replace("'", '').strip()
                        if not is_mcq_correct(ans, mcq_ans, mcq_options[mcq_ans].lower()):
                            local_dict["ans_idx"].append(iid+1)
                            mcq_error_count += 1
                            break
                    # go to human mode
                    else:
                        # directly false
                        if is_mcq_correct(ans, mcq_ans, mcq_options[mcq_ans].lower()):
                            continue
                        if is_mcq_direct_incrt(ans, mcq_ans):
                            local_dict["ans_idx"].append(iid+1)
                            mcq_error_count += 1
                            break
                        # not directly false, but is human evaled fasle
                        if not human_eval_mcq(output_dict["mcq"], ans, mcq_ans, mcq_options): # if model answer does not contain the correct option, go to human mode
                            local_dict["ans_idx"].append(iid+1)
                            mcq_error_count += 1
                            break
                        # else, treat as correct, continue
                else:
                    if not is_mcq_correct(ans, mcq_ans, mcq_options[mcq_ans].lower()):
                        local_dict["ans_idx"].append(iid+1)
                        mcq_error_count += 1
                        break
                    
        if local_dict["ans_idx"]:
            ret_list.append(local_dict)

    return ret_list, mcq_error_count


def find_yn_errors(output_list: list, target="all", global_yn_label="Yes", response_prefix='') -> list:
    assert target in ["all", "action", "place"]

    def is_yn_correct(model_ans, label_ans):
        return label_ans.lower() in model_ans.lower()
    
    def human_eval_yn(question, model_ans, label_ans):
        print(f"Question: {question}")
        print(f"Label answer: {label_ans}")
        print(f"Model answer: {model_ans}")
        input_res = input("Enter 0 for incorrect, 1 for correct: ")
        while input_res not in ['0', '1']:
            input_res = input("Invalid input. Please enter 0 for incorrect, 1 for correct: ")
        if input_res == '0':
            return False
        return True
    
    if target == "all":
        examine_list = output_list
    elif target == "action":
        examine_list = [output_dict for output_dict in output_list if output_dict["category"] == "action"]
    elif target == "place":
        examine_list = [output_dict for output_dict in output_list if output_dict["category"] == "place"]

    ret_list = []
    yn_error_count = 0
    for output_dict in examine_list:
        out_id = output_dict["id"]
        category = output_dict["category"]
        model_ans_lists = output_dict["yn_model_ans"]
        local_dict = {"id": out_id, "category": category, "ans_idx": []}
        for iid, ans_list in enumerate(model_ans_lists):
            for ans in ans_list:
                if response_prefix:
                    if response_prefix.lower() in ans.lower():
                        ans = ans.lower()
                        response_prefix = response_prefix.lower()
                        ans = ans.split(response_prefix)[-1].replace(':', '').replace('"', '').replace("'", '').strip()
                        if not is_yn_correct(ans, global_yn_label):   # only check for "Yes" answer by default
                            # print("YN Incorrct Model answer:", ans)
                            local_dict["ans_idx"].append(iid+1)
                            yn_error_count += 1
                            break
                    else:
                        if not human_eval_yn(output_dict["yn"], ans, global_yn_label):
                            local_dict["ans_idx"].append(iid+1)
                            yn_error_count += 1
                            break
                else:
                    if not is_yn_correct(ans, global_yn_label):
                        local_dict["ans_idx"].append(iid+1)
                        yn_error_count += 1
                        break
                     
        if local_dict["ans_idx"]:
            ret_list.append(local_dict)

    return ret_list, yn_error_count


def human_eval_sa_answers(output_list: list, split_symbol='') -> list:
    def check_sa_correctness_human(basic_str, iteration):
        print(basic_str)
        res_list = []
        for i in range(iteration):
            eval_result = input(f"Evaluating model answer {i+1}, enter 0 for incorrect, 1 for partially correct, 2 for correct: ")
            while eval_result not in ['0', '1', '2']:
                eval_result = input("Invalid input. Please enter 0 for incorrect, 1 for partially correct, 2 for correct: ")
            res_list.append(eval_result)
        return res_list
    
    evaler = input("Please enter your name in english: \n")
    eval_email = input("Please enter your email address: \n")
    start_time = datetime.datetime.now()

    total = len(output_list)
    ret_list = []
    for idx, output_dict in enumerate(output_list[:]):
        id = output_dict["id"]
        category = output_dict["category"]
        model_ans_lists = output_dict["sa_model_ans"]
        local_dict = {"id": id, "category": category, "image": [], "info": '',"human_eval": []}
        ans_str_list = []
        for iid, ans_list in enumerate(model_ans_lists):
            if split_symbol:
                ans_list = [ans.split(split_symbol)[-1].replace(':', '').replace('"', '').replace("'", '').strip() 
                            if split_symbol in ans else ans for ans in ans_list] # for cot only
            ans_set = list(set(ans_list))
            ans_string = f"[Model answer {iid+1}] {', '.join(ans_set)}"
            ans_str_list.append(ans_string)
        total_ans_str = '\n'.join(ans_str_list)
        # rebuild clean question string
        mcq_str = output_dict["mcq"]
        ques_str = "[Question] " + mcq_str.split("Question:")[1].split("Options:")[0].strip()
        ground_truth = "[Ground truth] " + list(output_dict["target"].values())[0]
        basics = f"{ques_str}\n{ground_truth}\n{total_ans_str}"
        num_ans = len(model_ans_lists)

        result = check_sa_correctness_human(basics, num_ans)
        
        local_dict["image"] = [i+1 for i in range(num_ans)]
        local_dict["info"] = basics
        local_dict["human_eval"] = result
        ret_list.append(local_dict)
        print(f"Progress: {idx+1}/{total} labeled.")
        print("\n")

    end_time = datetime.datetime.now()
    ret_list.append({"evaluator": evaler, "evaluator_email": eval_email, "start_time": str(start_time), "end_time": str(end_time), 
                     "time_used": str(end_time - start_time)})
    return ret_list


def find_sa_errors(evaled_ans_list: list, target='all') -> list:
    # human evaled, last dict is meta info
    if "evaluator" in evaled_ans_list[-1]:
        examine_list = evaled_ans_list[:-1]
    else:
        warnings.warn("No human evaluator info found.")
        examine_list = evaled_ans_list
    
    if target == 'all':
        pass
    elif target == 'action':
        examine_list = [item for item in examine_list if item["category"] == 'action']
    elif target == 'place':
        examine_list = [item for item in examine_list if item["category"] == 'place']

    ret_list = []
    sa_error_count = 0
    for eval_dict in examine_list:
        local_dict = {"id": eval_dict["id"], "category": eval_dict["category"], "ans_idx": []}
        for idx, ans in enumerate(eval_dict["human_eval"]):
            if ans == "0":
                local_dict["ans_idx"].append(idx+1)
                sa_error_count += 1
        if local_dict["ans_idx"]:
            ret_list.append(local_dict)
    return ret_list, sa_error_count


def batch_auto_plus_human_eval_ans(model_ans_paths: list, split_symbols: dict, error_stats_fp='mixed_error_stats.txt') -> list:
    def get_one_type_error(error_list: list, q_type: str) -> dict:
        one_type_error = []
        one_type_error_count = 0
        for error_dict in error_list:
            if error_dict["category"] == q_type:
                one_type_error.append(error_dict)
                one_type_error_count += len(error_dict["ans_idx"])
        return one_type_error, one_type_error_count
    
    
    for model_ans_path in model_ans_paths:
        model_ans_list = read_json(model_ans_path)
        # model_ans_list = model_ans_list[:5] + model_ans_list[-5:]
        
        model_name = os.path.basename(model_ans_path).split('_')[0]
        mcq_split_symbol = split_symbols["mcq"]
        yn_split_symbol = split_symbols["yn"]
        sa_split_symbol = split_symbols["sa"]
        
        mcq_error_list, mcq_error_count = find_mcq_errors(model_ans_list, response_prefix=mcq_split_symbol)
        yn_error_list, yn_error_count = find_yn_errors(model_ans_list, response_prefix=yn_split_symbol)
        write_json(f"{model_name}_mcq_err_dump.json", mcq_error_list)
        write_json(f"{model_name}_yn_err_dump.json", yn_error_list)
        evaled_error_list = human_eval_sa_answers(model_ans_list, sa_split_symbol)
        sa_out_path = os.path.basename(model_ans_path).replace('_outputs', '').replace('_updated', '').replace('.json', '_sa_human_eval_updated.json')
        write_json(sa_out_path, evaled_error_list)
        sa_error_list, sa_error_count = find_sa_errors(evaled_error_list)

        total_error_count = yn_error_count + mcq_error_count + sa_error_count
        total = get_total(model_ans_list)
        total_acc_percentage = (1 - total_error_count / (total * 3)) * 100
        yn_acc_percentage = (1 - yn_error_count / total) * 100
        mcq_acc_percentage = (1 - mcq_error_count / total) * 100
        sa_acc_percentage = (1 - sa_error_count / total) * 100

        action_yn_error_list, action_yn_error_count = get_one_type_error(yn_error_list, q_type='action')
        action_mcq_error_list, action_mcq_error_count = get_one_type_error(mcq_error_list, q_type='action')
        action_sa_error_list, action_sa_error_count = get_one_type_error(sa_error_list, q_type='action')
        
        action_total = get_total(model_ans_list, target='action')
        action_yn_acc_percentage = (1 - action_yn_error_count / action_total) * 100
        action_mcq_acc_percentage = (1 - action_mcq_error_count / action_total) * 100
        action_sa_acc_percentage = (1 - action_sa_error_count / action_total) * 100

        place_yn_error_list, place_yn_error_count = get_one_type_error(yn_error_list, q_type='place')
        place_mcq_error_list, place_mcq_error_count = get_one_type_error(mcq_error_list, q_type='place')
        place_sa_error_list, place_sa_error_count = get_one_type_error(sa_error_list, q_type='place')
        place_total = get_total(model_ans_list, target='place')
        
        place_yn_acc_percentage = (1 - place_yn_error_count / place_total) * 100
        place_mcq_acc_percentage = (1 - place_mcq_error_count / place_total) * 100
        place_sa_acc_percentage = (1 - place_sa_error_count / place_total) * 100

        print(f"Model answer file {os.path.basename(model_ans_path)}", file=open(error_stats_fp, 'a'))
        print(f"Model evaled file {os.path.basename(sa_out_path)}\n", file=open(error_stats_fp, 'a'))

        print(f"Model {model_name} total instances: {total}", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} action total instances: {action_total}", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} place total instances: {place_total}", file=open(error_stats_fp, 'a'))

        print(f"Model {model_name} mcq error instances: {mcq_error_count}", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} yn error instances: {yn_error_count}", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} sa error instances: {sa_error_count}", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} total error instances: {total_error_count}\n", file=open(error_stats_fp, 'a'))

        print(f"Model {model_name} total accuracy %: {total_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} yn accuracy %: {yn_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} mcq accuracy %: {mcq_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} sa accuracy %: {sa_acc_percentage:.1f}%\n", file=open(error_stats_fp, 'a'))

        print(f"Model {model_name} action yn accuracy %: {action_yn_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} action mcq accuracy %: {action_mcq_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} action sa accuracy %: {action_sa_acc_percentage:.1f}%\n", file=open(error_stats_fp, 'a'))

        print(f"Model {model_name} place yn accuracy %: {place_yn_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} place mcq accuracy %: {place_mcq_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} place sa accuracy %: {place_sa_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))

        # print(f"Model {model_name} error instances list: \n{mcq_error_list}\n{yn_error_list}\n{sa_error_list}\n", file=open(error_stats_fp, 'a'))

        print('='*40, file=open(error_stats_fp, 'a'))
        print('\n', file=open(error_stats_fp, 'a'))
    
    

def get_total(model_ans_list: list, target='all') -> int:
    assert target in ['all', 'action', 'place']
    if target == 'all':
        examine_list = model_ans_list
    elif target == 'action':
        examine_list = [item for item in model_ans_list if item["category"] == 'action']
    elif target == 'place':
        examine_list = [item for item in model_ans_list if item["category"] == 'place']
    total = 0
    for item in examine_list:
        total += len(item["sa_model_ans"])
    return total


def batch_find_and_print_error_stats(model_output_paths: list[str], sa_eval_paths: list[str], split_symbols={'mcq':'','yn':'','sa':''},
                                      error_stats_fp='error_stats.txt') -> None:
    def get_one_type_error(error_list: list, q_type: str) -> dict:
        one_type_error = []
        one_type_error_count = 0
        for error_dict in error_list:
            if error_dict["category"] == q_type:
                one_type_error.append(error_dict)
                one_type_error_count += len(error_dict["ans_idx"])
        return one_type_error, one_type_error_count
    
    with open(error_stats_fp, 'w') as f:
        pass
    
    for model_output_path, sa_eval_path in list(zip(model_output_paths, sa_eval_paths)):
        model_name = os.path.basename(model_output_path).split('_')[0]
        model_ans_list = read_json(model_output_path)
        evaled_error_list = read_json(sa_eval_path)

        mcq_split_symbol = split_symbols["mcq"]
        yn_split_symbol = split_symbols["yn"]
        sa_split_symbol = split_symbols["sa"]
        
        mcq_error_list, mcq_error_count = find_mcq_errors(model_ans_list, response_prefix=mcq_split_symbol)
        yn_error_list, yn_error_count = find_yn_errors(model_ans_list, response_prefix=yn_split_symbol)
        write_json(f"{model_name}_mcq_err_dump.json", mcq_error_list)
        write_json(f"{model_name}_yn_err_dump.json", yn_error_list)

        sa_error_list, sa_error_count = find_sa_errors(evaled_error_list)
        total_error_count = yn_error_count + mcq_error_count + sa_error_count
        total = get_total(model_ans_list)
        total_acc_percentage = (1 - total_error_count / (total * 3)) * 100
        yn_acc_percentage = (1 - yn_error_count / total) * 100
        mcq_acc_percentage = (1 - mcq_error_count / total) * 100
        sa_acc_percentage = (1 - sa_error_count / total) * 100

        action_yn_error_list, action_yn_error_count = get_one_type_error(yn_error_list, q_type='action')
        action_mcq_error_list, action_mcq_error_count = get_one_type_error(mcq_error_list, q_type='action')
        action_sa_error_list, action_sa_error_count = get_one_type_error(sa_error_list, q_type='action')
        
        action_total = get_total(model_ans_list, target='action')
        action_yn_acc_percentage = (1 - action_yn_error_count / action_total) * 100
        action_mcq_acc_percentage = (1 - action_mcq_error_count / action_total) * 100
        action_sa_acc_percentage = (1 - action_sa_error_count / action_total) * 100

        place_yn_error_list, place_yn_error_count = get_one_type_error(yn_error_list, q_type='place')
        place_mcq_error_list, place_mcq_error_count = get_one_type_error(mcq_error_list, q_type='place')
        place_sa_error_list, place_sa_error_count = get_one_type_error(sa_error_list, q_type='place')
        place_total = get_total(model_ans_list, target='place')
        
        place_yn_acc_percentage = (1 - place_yn_error_count / place_total) * 100
        place_mcq_acc_percentage = (1 - place_mcq_error_count / place_total) * 100
        place_sa_acc_percentage = (1 - place_sa_error_count / place_total) * 100

        print(f"Model answer file {os.path.basename(model_output_path)}", file=open(error_stats_fp, 'a'))
        print(f"Model evaled file {os.path.basename(sa_eval_path)}\n", file=open(error_stats_fp, 'a'))

        print(f"Model {model_name} total instances: {total}", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} action total instances: {action_total}", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} place total instances: {place_total}", file=open(error_stats_fp, 'a'))

        print(f"Model {model_name} mcq error instances: {mcq_error_count}", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} yn error instances: {yn_error_count}", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} sa error instances: {sa_error_count}", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} total error instances: {total_error_count}\n", file=open(error_stats_fp, 'a'))

        print(f"Model {model_name} total accuracy %: {total_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} yn accuracy %: {yn_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} mcq accuracy %: {mcq_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} sa accuracy %: {sa_acc_percentage:.1f}%\n", file=open(error_stats_fp, 'a'))

        print(f"Model {model_name} action yn accuracy %: {action_yn_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} action mcq accuracy %: {action_mcq_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} action sa accuracy %: {action_sa_acc_percentage:.1f}%\n", file=open(error_stats_fp, 'a'))

        print(f"Model {model_name} place yn accuracy %: {place_yn_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} place mcq accuracy %: {place_mcq_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))
        print(f"Model {model_name} place sa accuracy %: {place_sa_acc_percentage:.1f}%", file=open(error_stats_fp, 'a'))

        # print(f"Model {model_name} error instances list: \n{mcq_error_list}\n{yn_error_list}\n{sa_error_list}\n", file=open(error_stats_fp, 'a'))

        print('='*40, file=open(error_stats_fp, 'a'))
        print('\n', file=open(error_stats_fp, 'a'))


def build_path_list(model_names: list) -> tuple:
    obj_results = []
    sub_evaled_results = []
    for model_name in model_names:
        obj_results.append(model_name + '_outputs.json')
        sub_evaled_results.append(model_name + '_sa_human_eval.json')
    return obj_results, sub_evaled_results    


def merge_sub_eval_to_output(obj_res_paths: list, sub_res_paths: list, base_folder: str) -> None:
    assert len(sub_res_paths) == len(obj_res_paths), "Unequal path list length."
    for obj_path, sub_path in list(zip(obj_res_paths, sub_res_paths)):
        model_name = obj_path.split('_outputs')[0]
        obj_content_list = read_json(os.path.join(base_folder, obj_path))
        sub_content_list = read_json(os.path.join(base_folder, sub_path))
        for eval_dict in sub_content_list[:-1]:
            id = eval_dict["id"]
            category = eval_dict["category"]
            res = eval_dict["human_eval"]
            for i in range(len(obj_content_list)):
                if obj_content_list[i]["id"] == id and obj_content_list[i]["category"] == category:
                    obj_content_list[i]["sa_human_eval"] = res
                    break
        merged_dir = os.path.join(base_folder, f"{model_name}_evaled_outputs.json")
        write_json(merged_dir, obj_content_list)
        
        

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



def eval_mcq_ans_same_as_knowledge(model_ans_path: str, knowledge_ans_path: str) -> None:

    splits = {'mcq': 'answer is', 'yn': '', 'sa': ''}
    # batch_find_and_print_error_stats(model_output_paths=model_ans_paths, sa_eval_paths=sa_eval_paths, error_stats_fp='qwen-vl-chat_error_stats_vcd.txt')
    model_ans_list = read_json(model_ans_path)
    model_knowledge_list = read_json(knowledge_ans_path)
    total_question = get_total(model_ans_list, 'all')
    total_knowledge = get_total(model_knowledge_list, 'all')
    assert total_question == total_knowledge == 374, "Total question count mismatch"

    mcq_error_list, mcq_error_count = find_mcq_errors(model_ans_list)

    # qwen-vl-chat needs to read from the dump, it needs manual examination
    # mcq_error_list = read_json('/Users/xiaoyuan/Desktop/workspace/results_batch/updated_results/impr/cot/eval_results/qwen-vl-chat_mcq_err_dump.json')
    # mcq_error_count = sum([len(item["ans_idx"]) for item in mcq_error_list])

    knowledge_count = 0
    others_count = 0
    for error_dict in mcq_error_list:
        error_id = error_dict["id"]
        error_cat = error_dict["category"]
        for ans_dict in model_ans_list:
            if ans_dict["id"] == error_id and ans_dict["category"] == error_cat:
                for ans_idx in error_dict["ans_idx"]:
                    mcq_model_ans = ans_dict["mcq_model_ans"][ans_idx-1][0] # first repeat
                    for k_ans in model_knowledge_list: # find the corresponding model knowledge answer
                        if k_ans["id"] == error_id and k_ans["category"] == error_cat:
                            k_mcq_ans = k_ans["mcq_model_ans"][ans_idx-1][0] # first repeat
                            break
                    # print(f"Question: {ans_dict['mcq']}")
                    print(f"Correct answer: {ans_dict['MCQ_ans']} {ans_dict['MCQ_options'][ans_dict['MCQ_ans']]}")
                    print(f"Model answer: {mcq_model_ans}")
                    print(f"Model knowledge answer: {k_mcq_ans}")
                    _k_mcq_ans = k_mcq_ans.replace('.', '').lower()
                    _mcq_model_ans = mcq_model_ans.replace('.', '').lower()

                    # CHANGE HERE !!!
                    if _k_mcq_ans in _mcq_model_ans or _k_mcq_ans.split()[1] in _mcq_model_ans:
                    # if .startswith(_k_mcq_ans):
                    # if _mcq_model_ans.startswith(_k_mcq_ans):
                        knowledge_count += 1
                        error_dict["error_type"] = "knowledge"
                    else:
                        print('=================Notice above=================')
                        others_count += 1
                        error_dict["error_type"] = "other"
                    print('\n')
                break
    assert mcq_error_count == knowledge_count + others_count, "Count mismatch"
    print("Total error count: ", mcq_error_count)
    print("mcq error rate: ", round((mcq_error_count/total_question)*100, 1))
    print("knowledge error rate: ", round((knowledge_count/total_question)*100, 1))
    print("others error rate: ", round((others_count/total_question)*100, 1))

    model_name = os.path.basename(model_ans_path).split('_')[0]
    with open(f'{model_name}_mcq_error_dump.json', 'w') as f:
        json.dump(mcq_error_list, f)
    return mcq_error_count, knowledge_count, others_count



def two_open_answers_are_same(open_ans1: str, open_ans2: str) -> bool:
    print('='*40)
    print(f"[Model answer    ]: {open_ans1}")
    print(f"[Knowledge answer]: {open_ans2}")
    judegement = input("Enter 0 for different, 1 for same: ")
    while judegement not in ['0', '1']:
        judegement = input("Invalid input. Please enter 0 for different, 1 for same: ")
    return judegement == '1'



def get_knowledge_or_other_from_error_list(error_list: list, target_type: str):
    knowledge_count, other_count = 0, 0
    for error_dict in error_list:
        if error_dict["category"] == target_type:
            if error_dict["error_type"] == "knowledge":
                knowledge_count += len(error_dict["ans_idx"])
            elif error_dict["error_type"] == "other":
                other_count += len(error_dict["ans_idx"])
            else:
                raise ValueError("Invalid error type")
    return knowledge_count, other_count



def get_evaled_knowledge_or_other_from_error_list(error_list: list, target_type: str):
    knowledge_count, other_count = 0, 0
    for error_dict in error_list:
        if error_dict["category"] == target_type or target_type == 'all':
            if error_dict["similar"] == True:
                knowledge_count += len(error_dict["ans_idx"])
            else:
                other_count += len(error_dict["ans_idx"])
    return knowledge_count, other_count 



def human_eval_sa_answer_error_type_knowlegde_or_other(model_name, model_ans_path, sa_human_eval_path, knowledge_ans_path) -> list:
    sa_eval_list = read_json(sa_human_eval_path)
    model_ans_list = read_json(model_ans_path)
    model_knowledge_list = read_json(knowledge_ans_path)
    total_question = get_total(model_ans_list, 'all')

    sa_error_list, sa_error_count = find_sa_errors(sa_eval_list[:])
    sa_error_count = sum([len(item["ans_idx"]) for item in sa_error_list[:]])

    knowledge_count = 0
    others_count = 0
    error_dump = []
    for error_dict in sa_error_list:
        error_id = error_dict["id"]
        error_cat = error_dict["category"]
        for ans_dict in model_ans_list:
            if ans_dict["id"] == error_id and ans_dict["category"] == error_cat:
                for ans_idx in error_dict["ans_idx"]:
                    sa_model_ans = ans_dict["sa_model_ans"][ans_idx-1][0] # first repeat
                    for k_ans in model_knowledge_list: # find the corresponding model knowledge answer
                        if k_ans["id"] == error_id and k_ans["category"] == error_cat:
                            k_mcq_ans = k_ans["sa_model_ans"][ans_idx-1][0] # first repeat
                            break
                    # print(f"Question: {ans_dict['mcq']}")
                    print(f"Correct answer: {ans_dict['MCQ_options'][ans_dict['MCQ_ans']]}")
                    print(f"Model answer: {sa_model_ans}")
                    print(f"Model knowledge answer: {k_mcq_ans}")

                    if two_open_answers_are_same(sa_model_ans, k_mcq_ans):
                        local_dict = {"id": error_id, "category": error_cat, "ans_idx": [ans_idx], 
                                      "model_ans": sa_model_ans, "knowledge_ans": k_mcq_ans, "similar": True}
                        knowledge_count += 1
                    else:
                        local_dict = {"id": error_id, "category": error_cat, "ans_idx": [ans_idx], 
                                      "model_ans": sa_model_ans, "knowledge_ans": k_mcq_ans, "similar": False}
                        print('=================Notice above=================')
                        others_count += 1
                    error_dump.append(local_dict)
                    print('\n')
                break

    assert sa_error_count == knowledge_count + others_count, "Count mismatch"
    print("Total error count: ", sa_error_count)
    print("total error rate: ", round((sa_error_count/total_question)*100, 1))
    print("knowledge error rate: ", round((knowledge_count/total_question)*100))
    write_json(f'{model_name}_error_sa_types.json', error_dump)



def get_knowledge_vision_other_for_action_place(model_ans_path, model_knowledge_path, sa_evaled_path, target):
    my_target = 'place'

    model_names = ["llama3-llava-next-8b", "llava-v1.6-34b", "llava-1.5-13b", 
                   "blip2-t5-xxl", "instructblip-flan-t5-xxl", "qwen-vl",
                   "qwen-vl-chat", "gpt-4o-2024-05-13", "claude-3-5-sonnet-20240620"]
    
    model_name = "qwen-vl-chat"
    model_ans_path = f'/Users/xiaoyuan/Desktop/workspace/results_batch/updated_results/main/{model_name}_updated_outputs.json'
    model_knowledge_path = f'/Users/xiaoyuan/Desktop/workspace/results_batch/updated_results/sanity/{model_name}_with_uncertainty_new.json'

    # model_knowledge_path = f'/Users/xiaoyuan/Desktop/workspace/results_batch/updated_results/sanity/{model_name}_sanity_test.json'

    sa_evaled_fp = f"/Users/xiaoyuan/Desktop/workspace/results_batch/updated_results/sanity/eval_results/{model_name}_error_sa_types.json"


    model_ans_list = read_json(model_ans_path)
    type_total = get_total(model_ans_list, target=my_target)
    total = get_total(model_ans_list, target='all')
    
    # NEED TO CHANGE ASSERTION TYPES BELOW
    mcq_error_count_all, mcq_knowledge_count, _ = eval_mcq_ans_same_as_knowledge(model_ans_path, model_knowledge_path) # for sanity check, write to list
    read_path = os.path.basename(model_ans_path).split('_')[0] + '_mcq_error_dump.json'
    mcq_error_list = read_json(read_path)

    mc_knowledge, mc_other = get_knowledge_or_other_from_error_list(mcq_error_list, target_type=my_target)


    yn_error_list_all, yn_error_count_all = find_yn_errors(model_ans_list, target='all') # for sanity check
    yn_error_list, yn_error_count = find_yn_errors(model_ans_list, target=my_target)
    

    sa_evaled_list = read_json(sa_evaled_fp)
    sa_knowledge_error_all, sa_other_error_all = get_evaled_knowledge_or_other_from_error_list(sa_evaled_list, target_type='all')
    sa_knowledge_error, sa_other_error = get_evaled_knowledge_or_other_from_error_list(sa_evaled_list, target_type=my_target)
    
    print(f"YN acc: {round( (1 - yn_error_count_all/374) *100, 1) }")
    # print(f"YN {my_target} acc: {round( 1 - yn_error_count/type_total, 3) *100 }")
    print(f"MCQ acc: {round( (1 - mcq_error_count_all/374) *100, 1) }")
    print('MCQ knowledge count:', mcq_knowledge_count)
    # print(f"MCQ {my_target} acc: {round( 1 - mcq_error_count/type_total, 3) *100}")
    print(f"SA Knowledge: {round((sa_knowledge_error_all/374) *100, 1)}")
    print(f"SA Other: {round((sa_other_error_all/374) *100, 1) }")
    print('='*40)

    print(f"{my_target} YN knowledge error: {yn_error_count}")
    print(f"{my_target} YN other error: 0")

    print(f"{my_target} MCQ knowledge error: {mc_knowledge}")
    print(f"{my_target} MCQ other error: {mc_other}")

    print(f"{my_target} SA knowledge error: {sa_knowledge_error}")
    print(f"{my_target} SA other error: {sa_other_error}")

    print('='*40)
    print(f"{my_target} knowledge total: {yn_error_count+mc_knowledge+sa_knowledge_error}")
    print(f"{my_target} other total: {mc_other+sa_other_error}")
    print(f"{my_target} total: {type_total}, total*3: {type_total*3}")
    print(f"{my_target} knowledge percentage: {round((yn_error_count+mc_knowledge+sa_knowledge_error)/(type_total*3)*100, 1)}")
    print(f"{my_target} other percentage: {round((mc_other+0+sa_other_error)/(type_total*3)*100, 1)}")








if __name__ == "__main__":
    print("Hello, world!")
    # | ------------------------------------- |
    # |  PART 1: Eval and Collect Errors      |
    # | ------------------------------------- |
    # model_names = ["llama3-llava-next-8b", "llava-v1.6-34b", "llava-1.5-13b", 
    #                "blip2-t5-xxl", "instructblip-flan-t5-xxl", "qwen-vl",
    #                "qwen-vl-chat", "gpt-4o-2024-05-13", "claude-3-5-sonnet-20240620"]
    # model_names = ["llava-v1.6-34b", "llava-1.5-13b", "gpt-4o-2024-05-13"]
    # postfix_shorts = ['insist_csk', 'focus_vision']  
    # model_ans_path = '/Users/xiaoyuan/Desktop/workspace/results_batch/updated_results/main/llava-v1.6-34b_updated_outputs.json'
    # model_ans_list = read_json(model_ans_path)

    # mcq_error_list, mcq_error_count = find_mcq_errors(model_ans_list)
    # mcq_error_list_ac, mcq_error_count_ac = find_mcq_errors(model_ans_list, target='action')
    # mcq_error_list_pl, mcq_error_count_pl = find_mcq_errors(model_ans_list, target='place')

    # yn_error_list, yn_error_count = find_yn_errors(model_ans_list)
    # yn_error_list_ac, yn_error_count_ac = find_yn_errors(model_ans_list, target='action')
    # yn_error_list_pl, yn_error_count_pl = find_yn_errors(model_ans_list, target='place')

    # sa_human_eval_fp = '/Users/xiaoyuan/Desktop/workspace/results_batch/updated_results/main/llava-v1.6-34b_sa_human_eval_updated.json'
    # sa_evaled_list = read_json(sa_human_eval_fp)
    # sa_error_list, sa_error_count = find_sa_errors(sa_evaled_list)
    # sa_error_list_ac, sa_error_count_ac = find_sa_errors(sa_evaled_list, target='action')
    # sa_error_list_pl, sa_error_count_pl = find_sa_errors(sa_evaled_list, target='place')

    # total_question = get_total(model_ans_list, 'all')
    # action_question = get_total(model_ans_list, 'action')
    # place_question = get_total(model_ans_list, 'place')

    # # print(f"MCQ errors: {mcq_error_list}")

    # print(f"MCQ accuracy: {round((1 - mcq_error_count/total_question) *100, 1)}")
    # print(f"MCQ action accuracy: {round((1 - mcq_error_count_ac/action_question)*100, 1)}")
    # print(f"MCQ place accuracy: {round((1 - mcq_error_count_pl/place_question) *100, 1)}")

    # print(f"YN accuracy: {round((1 - yn_error_count/total_question) *100, 1)}")
    # print(f"YN action accuracy: {round((1 - yn_error_count_ac/action_question) *100, 1)}")
    # print(f"YN place accuracy: {round((1 - yn_error_count_pl/place_question) *100, 1)}")

    # print(f"SA accuracy: {round((1 - sa_error_count/total_question) *100, 1)}")
    # print(f"SA action accuracy: {round((1 - sa_error_count_ac/action_question) *100, 1)}")
    # print(f"SA place accuracy: {round((1 - sa_error_count_pl/place_question) *100, 1)}")

    # print(f"YN accuracy: {round(1 - yn_error_count/total_question, 3)}")


    # mcq_error_list = read_json('/Users/xiaoyuan/Desktop/workspace/results_batch/updated_results/impr/cot/eval_results/qwen-vl-chat_mcq_err_dump.json')
    # mcq_error_count = sum([len(item["ans_idx"]) for item in mcq_error_list])

    # print(f"MCQ errors: {mcq_error_list}")
    # batch_auto_plus_human_eval_ans(model_ans_paths=model_ans_paths, split_symbols=splits, error_stats_fp='qwen-vl-chat_error_stats_fov.txt')
    
    
    # | ------------------------------------- |
    # |  PART 2: Collect Error Statistics     |
    # | ------------------------------------- |
    # model_names = ["llama3-llava-next-8b", "llava-v1.6-34b", "llava-1.5-13b", 
    #                "blip2_t5_xxl", "instructblip-flan-t5-xxl", "qwen-vl",
    #                "qwen-vl-chat", "gpt-4o", "claude-3-5-sonnet"]
    
    # objective_paths, subjective_paths = build_path_list(model_names=model_names)
    # merge_sub_eval_to_output(obj_res_paths=objective_paths, sub_res_paths=subjective_paths, base_folder='results')
    # write_err_stats(obj_res_list=objective_paths, sub_res_list=subjective_paths, res_folder='results')

    # | ------------------------------------- |
    # |  PART 3: Construct Challenge Set      |
    # | ------------------------------------- |
    # result_folder = 'results'
    # image_input_folder =  "/120040051/test_resource/merged0728"
    # check_model = "qwen-vl"

    # # model name or global
    # lookup_file = os.path.join(result_folder, check_model+'_evaled_outputs.json')
    # err_stats_file = os.path.join(result_folder, 'error_stats.json')
    
    # # overwrite
    # challenge_set = 'challset_' + check_model 
    # challset_path = os.path.join(os.path.dirname(image_input_folder), challenge_set)
    # if os.path.exists(challset_path):
    #     shutil.rmtree(challset_path)
    # os.makedirs(challset_path)
    # print(f"Directory {challset_path} created")
    
    # err_stats = read_json(err_stats_file)
    # lookup_info = read_json(lookup_file)
    
    # target_error = "sa_error"
    # target_dict = err_stats[check_model]
    # target_errs = target_dict[target_error]
    # # target_err_mc = target_dict["mc_error"]
    # # target_err_sa = target_dict["sa_error"]
    # # target_err_all = target_dict["all_error"]
    
    # threshold = 1
    # filtered_errs = [item for item in target_errs if item[1] >= threshold]

    # challset_info = ret_input_info_upon_cond(lookup_list=lookup_info, cond_list=filtered_errs)
    # copy_imgs(challset_info, image_input_folder, challset_path)
    # out_info_path = os.path.join(challset_path, f'{check_model}_evaled_outputs_{target_error}_only.json')
    # write_json(out_info_path, challset_info)

    # print(f"error instances   (count >= {threshold}): {len(target_errs)}")
    # print(f"error instances % (count >= {threshold}): {len(target_errs)/344}")
    # print(f"error instances list: \n{target_errs}")