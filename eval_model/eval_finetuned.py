from transformers import LlavaForConditionalGeneration, AutoProcessor
from datasets import load_dataset
import torch, json, os
from tqdm import tqdm

def remove_label(complete_chat: str, response_str: str="ASSISTANT:") -> tuple:
    index = complete_chat.find(response_str)
    if index != -1:
        return complete_chat[:index + len(response_str)], complete_chat[index + len(response_str):]
    else:
        return complete_chat, ''
    
def eval_finetuned_model(model, processor, eval_dataset, output_dir):
    with open(output_dir, 'w') as f:
        for i in tqdm(range(len(eval_dataset)), desc="Generating answers"):
            _chat = processor.apply_chat_template(eval_dataset[i]["messages"], tokenize=False, add_generation_prompt=False)
            text_prompt, label = remove_label(_chat)
            inputs = processor(text=text_prompt, images=eval_dataset[i]["images"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, do_sample=False, max_new_tokens=25)
            generated_text = processor.decode(outputs[0], skip_special_tokens=False)
            model_answer = generated_text.split('ASSISTANT:')[1]
            json.dump({'question': text_prompt, 'label': label, 'answer': generated_text}, f)
            f.write('\n')
            f.flush()


def detect_errors_ft(res_fp, error_fp='ft_error_stats.txt'):
    with open(res_fp, 'r') as f:
        dict_list = []
        for line in f:
            info_dict = json.loads(line)
            dict_list.append(info_dict)

    total_single_category = len(dict_list) / 4
    total_all_categories = len(dict_list) / 4 * 3

    total_wrong, total_correct = 0, 0

    yn_correct, yn_action_correct, yn_place_correct = 0, 0, 0
    mcq_correct, mcq_action_correct, mcq_place_correct = 0, 0, 0
    sa_correct, sa_action_correct, sa_place_correct = 0, 0, 0

    category = 'action'
    for info_dict in dict_list:
        question = info_dict['question'].strip()
        label = info_dict['label'].strip()
        answer = info_dict['answer'].split('ASSISTANT:')[1].strip()
        if "Question: Where" in question:
            if category == 'action':
                category = 'place'

        if label in ['Yes', 'No']:
            if label == 'No':
                continue
            correctness = 2 if answer.startswith(label) else 0
            yn_correct += 1 if correctness == 2 else 0
            if category == 'action':
                yn_action_correct += 1 if correctness == 2 else 0
            else:
                yn_place_correct += 1 if correctness == 2 else 0

        elif label in ['A', 'B', 'C', 'D']:
            correctness = 2 if (answer.startswith(label) or answer.startswith(f"({label}")) else 0
            mcq_correct += 1 if correctness == 2 else 0
            if category == 'action':
                mcq_action_correct += 1 if correctness == 2 else 0
            else:
                mcq_place_correct += 1 if correctness == 2 else 0
        else:
            print(f"### Question: {question}")
            print(f"### Label: {label}")
            print(f"### Model Ans: {answer}")
            correctness = input('\nEnter the judgement: 0 incorrect, 1 partially correct, 2 correct: ')
            sa_correct += 1 if int(correctness) != 0 else 0
            if category == 'action':
                sa_action_correct += 1 if int(correctness) != 0 else 0
            else:
                sa_place_correct += 1 if int(correctness) != 0 else 0
        if int(correctness) == 0:
            total_wrong += 1

    print(f'Evaluating {os.path.basename(res_fp)}', file=open(error_fp, 'w'))
    print('='*50, file=open(error_fp, 'a'))
    print()

    print(f'Total (all): {total_all_categories}', file=open(error_fp, 'a'))
    print(f'Total (MC/YN/SA): {total_single_category}', file=open(error_fp, 'a'))
    print(f'Total Wrong: {total_wrong}', file=open(error_fp, 'a'))
    print(f'Accuracy: {1 - total_wrong/total_all_categories}', file=open(error_fp, 'a'))
    # print(f'Correct: {correct}')

    print(f"mcq_correct: {mcq_correct}", file=open(error_fp, 'a'))
    print(f"yn_correct: {yn_correct}", file=open(error_fp, 'a'))
    print(f"sa_correct: {sa_correct}", file=open(error_fp, 'a'))
    print(f'MCQ Accuracy: {mcq_correct/(total_single_category)}', file=open(error_fp, 'a'))
    print(f'YN Accuracy: {yn_correct/total_single_category}', file=open(error_fp, 'a'))
    print(f'SA Accuracy: {sa_correct/total_single_category}', file=open(error_fp, 'a'))

    action_total_single = 35
    place_total_single = 41

    print(f'MCQ Action Accuracy: {mcq_action_correct/(action_total_single)}', file=open(error_fp, 'a'))
    print(f'YN Action Accuracy: {yn_action_correct/(action_total_single)}', file=open(error_fp, 'a'))
    print(f'SA Action Accuracy: {sa_action_correct/(action_total_single)}', file=open(error_fp, 'a'))
    print(f'MCQ Place Accuracy: {mcq_place_correct/(place_total_single)}', file=open(error_fp, 'a'))
    print(f'YN Place Accuracy: {yn_place_correct/(place_total_single)}', file=open(error_fp, 'a'))
    print(f'SA Place Accuracy: {sa_place_correct/(place_total_single)}', file=open(error_fp, 'a'))        



if __name__ == "__main__":
    model_dir = '/120040051/MLLM_Repos/llava-1.5-13b-ft/checkpoint-54'
    llava_model = LlavaForConditionalGeneration.from_pretrained(model_dir, device_map='auto').eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    eval_dataset = load_dataset('xiaoyuanliu/conflict_vis', split='test')
    output_dir = '/120040051/Github_Repos/VKConflict/eval_model/updated_results/llava-1.5-13b-ft_updated_outputs.json'
    eval_finetuned_model(llava_model, processor, eval_dataset, output_dir)