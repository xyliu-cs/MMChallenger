import os
import json


def identify_model_type(answer_file_path):
    base_name = os.path.basename(answer_file_path)
    if "llava15" in base_name:
        return "llava-1.5"
    elif "instructblip" in base_name:
        return "instructblip"
    else:
        raise KeyError (f"Unsupported model answer {base_name}")


def shorten_error_list(typed_error_dict):
    short_list = []
    for id, key_list in typed_error_dict.items():
        img1, img2 = 0, 0
        for key_name in key_list:
            if "img1" in key_name:
                img1 += 1
            elif "img2" in key_name:
                img2 += 1
        if img1 != 0:
            short_list.append(f"{str(id)}a({img1})")
        if img2 != 0:
            short_list.append(f"{str(id)}b({img2})")
    return short_list


def type_and_shorten(error_dict, _key):
    typed_error = {}
    for qid, error_list in error_dict.items():
        if len(error_list[_key]) != 0:
            typed_error[qid] = error_list[_key]
    return shorten_error_list(typed_error)



def construct_result_table(input_file_path, model_type, option_num, iter=2, repeat=5):
    error = {}
    choice_success = {}
    unexpected = {}
    option_list = [chr(i) for i in range(ord('A'), ord('A') + option_num)]

    with open(input_file_path, 'r') as f:
        answer_list = json.load(f)
    total_len = len(answer_list) * 2
    for answer in answer_list:
        q_id = answer['id']
        dt_option = answer['distractor1 answer']  # A
        gt_option = answer['choice answer']       # A
        yes_ans = answer['binary-yes answer']
        no_ans = answer['binary-no answer']
        cp_ans = answer['binary-cp answer']

        if model_type == "instructblip":
            gt_option = f"({gt_option})"
            dt_option = f"({dt_option})"
    
        local_error = {"choice":[], "binary-yes":[], "binary-no":[], "binary-cp":[]}
        local_unexp = {"choice":[], "binary-yes":[], "binary-no":[], "binary-cp":[]}
        local_choice_success = [] # failure on distractor
        
        for i in range(iter):
            for j in range(repeat):
                choice_key = f"choice_ans_img{i+1}_{j+1}"
                choice_answer_text = answer[choice_key]
                if not choice_answer_text.startswith(gt_option): # error
                    if choice_answer_text.startswith(dt_option): # success
                        local_choice_success.append(choice_key)
                        local_error['choice'].append(choice_key)
                    else:
                        not_start_letter = True
                        for letter in option_list:
                            if choice_answer_text.startswith(letter):
                                not_start_letter = False
                                break
                        if not_start_letter:                    # unexpected
                            local_unexp['choice'].append(choice_key)
                        else:                 
                            local_error['choice'].append(choice_key)
            
                binary_yes_key = f"binary-yes_ans_img{i+1}_{j+1}"
                binary_yes_answer = answer[binary_yes_key]
                if not binary_yes_answer.startswith(yes_ans):  # error
                    if binary_yes_answer.startswith(no_ans):   # success
                        local_error['binary-yes'].append(binary_yes_key)
                    else:                                      # unexpected
                        local_unexp['binary-yes'].append(binary_yes_key)

                binary_no_key = f"binary-no_ans_img{i+1}_{j+1}"
                binary_no_answer = answer[binary_no_key]
                if not binary_no_answer.startswith(no_ans):
                    if binary_no_answer.startswith(yes_ans):
                        local_error['binary-no'].append(binary_no_key)
                    else:
                        local_unexp['binary-no'].append(binary_no_key)

                binary_cp_key = f"binary-cp_ans_img{i+1}_{j+1}"
                binary_cp_answer = answer[binary_cp_key]
                if not binary_cp_answer.startswith(cp_ans):
                    if binary_cp_answer.startswith(yes_ans):
                        local_error['binary-cp'].append(binary_cp_key)
                    else:
                        local_unexp['binary-cp'].append(binary_cp_key)
        error[q_id] = local_error
        unexpected[q_id] = local_unexp
        choice_success[q_id] = local_choice_success
    
    key_pool = ['choice', 'binary-yes', 'binary-no', 'binary-cp']
    res_dict = {}
    for i in key_pool:
        res_dict[i] = type_and_shorten(error, i)
    res_dict['choice_success'] = shorten_error_list(choice_success) # already typed

    return total_len, res_dict




def show_results(total_len, res_dict, dump=False):
    for question_type in res_dict:
        error_list = res_dict[question_type]
        print('='*40)
        print(question_type)
        print('Error cases: ' + ', '.join(error_list))
        print(f"Total error: {len(error_list)}, Error trigger rate = {round(len(error_list)/total_len, 2)}")

    if dump:
        with open('./error_results_dump.txt', 'w') as f:
            for question_type in res_dict:
                error_list = res_dict[question_type]
                f.write('='*40 + '\n')
                f.write(question_type + '\n')
                f.write('Error cases: ' + ', '.join(error_list) + '\n')
                f.write(f"Total error: {len(error_list)}, Error trigger rate = {round(len(error_list)/total_len, 2)}\n")




if __name__ == '__main__':
    input_file_path = "/120040051/test_resource/output_answers/location_answers_0329_instructblip-7b.json"
    model_name = identify_model_type(input_file_path)
    print(f"Checking answers from {model_name} ...")
    total_test_case, error_dict = construct_result_table(input_file_path, model_name, 4)
    show_results(total_test_case, error_dict, dump=False)