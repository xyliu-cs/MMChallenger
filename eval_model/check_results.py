import os
import json
from collections import defaultdict


def identify_model_type(answer_file_path):
    base_name = os.path.basename(answer_file_path)
    if "llava15" in base_name:
        return "llava-1.5"
    elif "instructblip" in base_name:
        return "instructblip"
    else:
        raise KeyError (f"Unsupported model answer {base_name}")


def construct_result_table(json_data, model_type, option_num, img_per_content=2, repeat=5):

    option_list = [chr(i) for i in range(ord('A'), ord('A') + option_num)]

    error_full = {"choice":defaultdict(list), "binary-yes":defaultdict(list), "binary-no":defaultdict(list), "binary-cp":defaultdict(list)}
    error_short = {"choice":set(), "binary-yes":set(), "binary-no":set(), "binary-cp":set()}

    unexpected_full = {"choice":defaultdict(list), "binary-yes":defaultdict(list), "binary-no":defaultdict(list), "binary-cp":defaultdict(list)}
    unexpected_short = {"choice":set(), "binary-yes":set(), "binary-no":set(), "binary-cp":set()}

    choice_success_short = set()   # failure on distractor

    for answer in json_data:
        q_id = answer['id']
        dt_option = answer['distractor1 answer']  # A
        gt_option = answer['choice answer']       # A
        yes_ans = answer['binary-yes answer']
        no_ans = answer['binary-no answer']
        cp_ans = answer['binary-cp answer']

        if model_type == "instructblip":
            gt_option = f"({gt_option})"
            dt_option = f"({dt_option})"
            option_list = [f'({option})' for option in option_list]
    
        for i in range(img_per_content):
            for j in range(repeat):
                short_name = f"q{q_id}_img{i+1}"

                choice_key = f"choice_ans_img{i+1}_{j+1}"
                # choice_key = f"choice1_ans_empty_{j+1}"
                choice_answer_text = answer[choice_key]
                if not choice_answer_text.startswith(gt_option): # error
                    if choice_answer_text.startswith(dt_option): # success
                        choice_success_short.add(short_name)
                        error_full['choice'][q_id].append(choice_key)
                        error_short['choice'].add(short_name)
                    else:
                        not_start_letter = True
                        for letter in option_list:
                            if choice_answer_text.startswith(letter):
                                not_start_letter = False
                                break
                        if not_start_letter:                    # unexpected
                            unexpected_full['choice'][q_id].append(choice_key)
                            unexpected_short['choice'].add(short_name)

                        else:                                   # false positive
                            error_full['choice'][q_id].append(choice_key)
                            error_short['choice'].add(short_name)
            

                binary_yes_key = f"binary-yes_ans_img{i+1}_{j+1}"
                # binary_yes_key = f"binary-yes_ans_empty_{j+1}"
                binary_yes_answer = answer[binary_yes_key]
                if not binary_yes_answer.startswith(yes_ans):  # error
                    if binary_yes_answer.startswith(no_ans):   # success
                        error_full['binary-yes'][q_id].append(binary_yes_key)
                        error_short['binary-yes'].add(short_name)
                    else:                                      # unexpected
                        error_full['binary-yes'][q_id].append(binary_yes_key)
                        unexpected_short['binary-yes'].add(short_name)


                binary_no_key = f"binary-no_ans_img{i+1}_{j+1}"
                # binary_no_key = f"binary-no_ans_empty_{j+1}"
                binary_no_answer = answer[binary_no_key]
                if not binary_no_answer.startswith(no_ans):
                    if binary_no_answer.startswith(yes_ans):
                        error_full['binary-no'][q_id].append(binary_no_key)
                        error_short['binary-no'].add(short_name)
                    else:
                        unexpected_full['binary-no'][q_id].append(binary_no_key)
                        unexpected_short['binary-no'].add(short_name)


                binary_cp_key = f"binary-cp_ans_img{i+1}_{j+1}"
                # binary_cp_key = f"binary-cp_ans_empty_{j+1}"
                binary_cp_answer = answer[binary_cp_key]
                if not binary_cp_answer.startswith(cp_ans):
                    if binary_cp_answer.startswith(yes_ans):
                        error_full['binary-cp'][q_id].append(binary_cp_key)
                        error_short['binary-cp'].add(short_name)
                    else:
                        unexpected_full['binary-cp'][q_id].append(binary_cp_key)
                        unexpected_short['binary-cp'].add(short_name)


    return choice_success_short, error_full, error_short, unexpected_full, unexpected_short


if __name__ == '__main__':
    input_file_path = "/120040051/test_resource/output_answers/location_answers_NT_0502_llava15-7b.json"
    model_name = identify_model_type(input_file_path)
    print(f"Checking answers from {model_name} ...")

    with open(input_file_path, 'r') as f:
        answers = json.load(f)

    total_content = len(answers)
    image_per_question = 2
    question_per_type = total_content * image_per_question

    mcq_success, error_full_table, error_short_table, unexp_full_table, unexp_short_table = construct_result_table(answers, model_name, option_num=4, img_per_content=2, repeat=5)

    print("="*40)
    print("Summary")
    print("="*40)

    print(f"MCQ Total: {question_per_type}")
    print(f"MCQ Success: {len(mcq_success)}")
    print(f"MCQ Error: {len(error_short_table['choice'])}")
    print(f"MCQ Unexpected: {len(unexp_short_table['choice'])}")
    print(f"Error rate = {len(mcq_success)/question_per_type}")
    print(f"False positive rate = {(len(error_short_table['choice']) - len(mcq_success))/question_per_type}")
    print(f"Unexpected rate = {len(unexp_short_table['choice'])/question_per_type}")
    print()

    print(f"Binary Total: {question_per_type}")
    print(f"Binary Error: {len(error_short_table['binary-no'])}")
    print(f"Binary Unexpected: {len(unexp_short_table['binary-no'])}")
    print(f"Error rate = {len(error_short_table['binary-no'])/question_per_type}")
    print(f"Unexpected rate = {len(unexp_short_table['binary-no'])/question_per_type}")