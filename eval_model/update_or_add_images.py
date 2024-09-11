import filecmp
import json, os, sys, copy
from PIL import Image
import shutil
from eval import eval_model
from detect_errors import eval_sa_answers


def replace_images(update_image_folder: str, original_image_folder: str, image_count: int=14):
    update_folder = os.path.normpath(update_image_folder)
    original_folder = os.path.normpath(original_image_folder)
    image_names = os.listdir(update_folder)
    assert len(image_names) == 14, f"Number of images found in the update folder = {len(image_names)} is not equal to expected number {image_count}"
    deprecated_folder = os.path.join(original_folder, "deprecated")
    os.makedirs(deprecated_folder, exist_ok=True)
    for image_name in image_names:
        deprecated_image_path = os.path.join(deprecated_folder, image_name)
        ori_image_path = os.path.join(original_folder, image_name)
        update_image_path = os.path.join(update_folder, image_name)
        
        if image_name in os.listdir(original_folder):
            # move the original image to the deprecated folder
            if os.path.exists(deprecated_image_path):
                if filecmp.cmp(ori_image_path, deprecated_image_path):
                    print(f"Image {image_name} is already present in the deprecated folder and is identical to the one in the original folder, replacing the one in the original folder AGAIN with the new one")
                    shutil.copy(update_image_path, ori_image_path)
                    continue
                # else, already moved, just pass
                else:
                    print(f"Image {image_name} is already present in the deprecated folder and is different from the one in the original folder, no operation is needed")
                    continue
            else:
                shutil.move(ori_image_path, deprecated_image_path)
                print(f"Image {image_name} is moved to the deprecated folder successfully")
        else:
            raise FileNotFoundError(f"Image {image_name} not found in the original folder")
        
        # copy the new image to the original folder
        shutil.copy(update_image_path, ori_image_path)
        print(f"Image {image_name} is copied to {update_image_path} successfully")
    print(f"All {len(image_names)} images are updated successfully")


def create_tmp_input_file(image_folder: str, question_file: str, output_file: str):
    image_folder = os.path.normpath(image_folder)
    question_file = os.path.normpath(question_file)
    output_file = os.path.normpath(output_file)
    
    with open(question_file, 'r') as f:
        question_data = json.load(f)
        
    output_list = []
    for image_name in os.listdir(image_folder):
        # print(image_name)
        found = False
        for question_dict in question_data:
            image_info_list = question_dict["image"]
            # print(image_info_list)
            if image_name in image_info_list:
                output_list.append(question_dict)
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Image {image_name} not found in the question file")
    with open(output_file, 'w') as f:
        json.dump(output_list, f, indent=2)
    print(f"Update input file is created successfully at {output_file}")


def batch_eval(model_dirs, model_types, input_file_path, image_folder, output_folder):
    for model_dir, model_type in list(zip(model_dirs, model_types))[1:]:
        model_name = os.path.basename(model_dir)
        print('Evaluating model:', model_name)
        eval_model(model_dir, model_type, input_file_path, image_folder, f"{output_folder}/{model_name}_output_update_batch14.json")
        print('Finished evaluating model:', model_name)


def batch_merge_to_original_ans(original_ans_files, update_ans_files, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for original_ans_file, update_ans_file in list(zip(original_ans_files, update_ans_files)):
        print(f"original_ans_file: {original_ans_file}")
        print(f"Updating answer file: {update_ans_file}")
        with open(original_ans_file, 'r') as f:
            original_ans = json.load(f)
        with open(update_ans_file, 'r') as f:
            update_ans = json.load(f)
        
        for ans_dict in update_ans:
            prime_key_1 = ans_dict["id"]
            prime_key_2 = ans_dict["category"]
            found = False
            for i, ori_dict in enumerate(original_ans):
                if ori_dict["id"] == prime_key_1 and ori_dict["category"] == prime_key_2:
                    original_ans[i] = ans_dict # update the answer
                    found = True
                    break
            assert found, f"Answer not found for id: {prime_key_1} and category: {prime_key_2}"
        
        out_file = f"{output_folder}/{os.path.basename(original_ans_file).replace('outputs', 'updated_outputs')}"
        
        with open(out_file, 'w') as f:
            json.dump(original_ans, f, indent=2)
        print(f"Updated answer file is created successfully at {out_file}")
    print(f"All {len(original_ans_files)} answer files are updated successfully")


def batch_eval_new_open_answers(update_ans_files, original_sa_eval_files, update_file_folder, compare_diff_first=False):
    for update_ans_file, original_sa_eval_file in list(zip(update_ans_files, original_sa_eval_files)):
        print(f"original_ans_file: {original_sa_eval_file}")
        print(f"Updating answer file: {update_ans_file}")
        with open(update_ans_file, 'r') as f:
            update_ans = json.load(f)
        with open(original_sa_eval_file, 'r') as f:
            original_sa_eval = json.load(f)
            
        sa_evaled_ans = eval_sa_answers(update_ans)
        for eval_dict in sa_evaled_ans[:-1]:
            prime_key_1 = eval_dict["id"]
            prime_key_2 = eval_dict["category"]
            found = False
            for i, ori_dict in enumerate(original_sa_eval):
                if ori_dict["id"] == prime_key_1 and ori_dict["category"] == prime_key_2:
                    original_sa_eval[i] = eval_dict # update the answer
                    found = True
                    break
            if not found:
                raise ValueError(f"Human evaluation not found for id: {prime_key_1} and category: {prime_key_2}")
        
        out_file = f"{update_file_folder}/{os.path.basename(original_sa_eval_file).replace('eval', 'eval_updated')}"
        with open(out_file, 'w') as f:
            json.dump(original_sa_eval, f, indent=2)


def batch_eval_new_open_answers_from_raw(update_ans_files, original_ans_files, original_sa_eval_files, update_file_folder, file_name):
    for update_ans_file, original_ans_file, original_sa_eval_file in list(zip(update_ans_files, original_ans_files, original_sa_eval_files)):
        print(f"original_ans_file: {original_ans_file}")
        print(f"Updating answer file: {update_ans_file}")
        with open(update_ans_file, 'r') as f:
            update_ans = json.load(f)
        with open(original_sa_eval_file, 'r') as f:
            original_sa_eval = json.load(f)
        with open(original_ans_file, 'r') as f:
            original_ans = json.load(f)

        out_file = f"{update_file_folder}/{file_name}"
        with open(out_file, 'w') as f:
            json.dump([{'1': '0'}], f, indent=2)

        # find the differences first
        diff_ans_to_eval = []
        for eval_dict in update_ans:
            prime_key_1 = eval_dict["id"]
            prime_key_2 = eval_dict["category"]
            flattened_list = [item for sublist in eval_dict["sa_model_ans"] for item in sublist]
            update_ans = set(flattened_list)
            # update_ans = set(eval_dict["sa_model_ans"])
            found = False
            for i, ori_dict in enumerate(original_ans):
                if ori_dict["id"] == prime_key_1 and ori_dict["category"] == prime_key_2:
                    found = True
                    flattened = [item for sublist in ori_dict["sa_model_ans"] for item in sublist]
                    ori_ans = set(flattened)
                    if ori_ans != update_ans:
                        diff_ans_to_eval.append(eval_dict)
                    break
            if not found:
                raise ValueError(f"#1 Human evaluation item not found for id: {prime_key_1} and category: {prime_key_2}")            
        
        # eval the differences
        print(len(diff_ans_to_eval))
        sa_evaled_ans = eval_sa_answers(diff_ans_to_eval)
        
        # use the difference info to update the original_sa_eval
        for eval_dict in sa_evaled_ans[:-1]:
            prime_key_1 = eval_dict["id"]
            prime_key_2 = eval_dict["category"]
            found = False
            for i, ori_dict in enumerate(original_sa_eval):
                if ori_dict["id"] == prime_key_1 and ori_dict["category"] == prime_key_2:
                    original_sa_eval[i] = eval_dict # update the answer
                    found = True
                    break
            if not found:
                raise ValueError(f"#2 Human evaluation item not found for id: {prime_key_1} and category: {prime_key_2}")
        
        # write the output info
        out_file = f"{update_file_folder}/{file_name}"
        with open(out_file, 'w') as f:
            json.dump(original_sa_eval, f, indent=2)
   
if __name__ == "__main__":
    # update_image_folder = '/120040051/test_resource/update_batch'
    # original_image_folder = '/120040051/test_resource/merged0728'
    # # replace_images(update_image_folder, original_image_folder, 14)
    # # create_tmp_input_file(update_image_folder, '/120040051/test_resource/merged0728/input_info.json', '/120040051/test_resource/merged0728/input_update_info.json')
    
    # model_dirs = ['/120040051/MLLM_Repos/llava-1.5-13b-hf', '/120040051/MLLM_Repos/blip2-flan-t5-xxl', '/120040051/MLLM_Repos/Qwen-VL', '/120040051/MLLM_Repos/llama3-llava-next-8b-hf', 
    #               '/120040051/MLLM_Repos/instructblip-flan-t5-xxl', '/120040051/MLLM_Repos/Qwen-VL-Chat', '/120040051/MLLM_Repos/llava-v1.6-34b-hf']
    # model_types = ["llava-vicuna", "blip2-t5", "qwen-vl", "llava-llama3", "instructblip-t5", "qwen-vl", "llava-yi"]
    
    # output_folder = '/120040051/test_resource/output_answers'
    # # batch_eval(model_dirs, model_types, '/120040051/test_resource/merged0728/input_update_info.json', original_image_folder, output_folder)
    
    # # these names should correspond the model_dirs !!!
    # model_names = [ "llava-1.5-13b", "blip2_t5_xxl", "qwen-vl", 
    #              "llama3-llava-next-8b", "instructblip-flan-t5-xxl", "qwen-vl-chat", "llava-v1.6-34b"]
    # ori_folder = "/120040051/Github_Repos/VKConflict/eval_model/results"
    # original_answer_files = [f"{os.path.join(ori_folder, model_name)}_outputs.json" for model_name in model_names]
    # original_sa_eval_files = [f"{os.path.join(ori_folder, ori_name)}_sa_human_eval.json" for ori_name in model_names]
    # update_files = [f"{output_folder}/{os.path.basename(model_name)}_output_update_batch14.json" for model_name in model_dirs]
    update_output_folder = "/120040051/Github_Repos/VKConflict/eval_model/updated_results"
    # batch_merge_to_original_ans(original_answer_files, update_files, update_output_folder)
    # batch_eval_new_open_answers(update_files, original_sa_eval_files, update_output_folder)
    
    vcd_update = ['/120040051/Github_Repos/VKConflict/eval_model/updated_results/llava-1.5_layers_2-32_tokens_25_eos_attn_0.5_cfg_2.0.json']
    original = ['/120040051/Github_Repos/VKConflict/eval_model/updated_results/llava-1.5_layers_2-32_tokens_25_eos_attn_0.5_cfg_1.1.json']
    original_eval = ['/120040051/Github_Repos/VKConflict/eval_model/updated_results/llava-1.5-13b_sa_human_eval_updated_layers_2-32_tokens_25_eos_attn_0.5_cfg_1.1.json']
    file_name = 'llava-1.5-13b_sa_human_eval_updated_layers_2-32_tokens_25_eos_attn_0.5_cfg_2.0.json'
    batch_eval_new_open_answers_from_raw(vcd_update, original, original_eval, update_output_folder, file_name)