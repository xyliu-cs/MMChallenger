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
                        if is_mcq_correct(ans, mcq_ans):
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
                    if response_prefix in ans:
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




if __name__ == "__main__":

    # | ------------------------------------- |
    # |  PART 1: Eval and Collect Errors      |
    # | ------------------------------------- |
    # model_names = ["llama3-llava-next-8b", "llava-v1.6-34b", "llava-1.5-13b", 
    #                "blip2-t5-xxl", "instructblip-flan-t5-xxl", "qwen-vl",
    #                "qwen-vl-chat", "gpt-4o-2024-05-13", "claude-3-5-sonnet-20240620"]
    # model_names = ["llava-v1.6-34b", "llava-1.5-13b", "gpt-4o-2024-05-13"]
    # postfix_shorts = ['insist_csk', 'focus_vision']  

    model_ans_paths = ['/120040051/Github_Repos/VKConflict/eval_model/updated_results/qwen-vl_chat_outputs_vcd_a1_b01_n500_chat_fmt.json']
    sa_eval_paths = ['/120040051/Github_Repos/VKConflict/eval_model/updated_results/qwen-vl_chat_outputs_vcd_a1_b01_n500_chat_fmt_sa_human_eval_updated.json']
    splits = {'mcq': '', 'yn': '', 'sa': ''}
    batch_find_and_print_error_stats(model_output_paths=model_ans_paths, sa_eval_paths=sa_eval_paths, error_stats_fp='qwen-vl-chat_error_stats_vcd.txt')
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