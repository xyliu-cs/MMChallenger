import random, os, json

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)
    print(f"Write {len(data)} items to {file_path}")


def select_items_from_list(lst):
    # Set a seed for reproducibility, if necessary
    random.seed(1024)
    # Define the segments: top 5, middle 5, and last 5
    top_five = lst[:5]
    middle_index = len(lst) // 2
    middle_five = lst[max(middle_index - 2, 0):min(middle_index + 3, len(lst))]
    last_five = lst[-5:]
    # Randomly select one item from each segment
    top = random.choice(top_five)
    middle = random.choice(middle_five)
    last = random.choice(last_five)
    return top, middle, last


def underscore(string: str) -> str:
    return string.replace(' ', '_')


def find_candidate_list_asc(complete_data_frame, category, subject, action, place):
    if category == "place":
        ret = []
        for small_dict in complete_data_frame:
            if small_dict["category"] == 0:
                if small_dict["subject"] == subject and small_dict["action"] == action:
                    # omit gt
                    if small_dict["place"] != place: 
                        ret.append([small_dict["place"], small_dict["logprob"]])
    elif category == "action":
        ret = []
        for small_dict in complete_data_frame:
            if small_dict["category"] == 0:
                if small_dict["subject"] == subject and small_dict["place"] == place:
                    # omit gt
                    if small_dict["action"] != action: 
                        ret.append([small_dict["action"], small_dict["logprob"]])        
    return list(sorted(ret, key=lambda x: x[1], reverse=False))
        

# the gt target should be removed from the candidate list
# we only keep the necessary info in the input json
# gt triplets always [subject, action, place]
def generate_text_inputs(gt_index: int, gt_triplet: list, gt_logprob: float, candidiate_list_asc: list, category: str, option_seed, image_fmt='jpg') -> dict:
    if category == "place":
        subject, action, place = gt_triplet[0], gt_triplet[1], gt_triplet[2]
        _subject, _action, _place = underscore(subject), underscore(action), underscore(place)
        option_1_tup, option_2_tup, option_3_tup = select_items_from_list(candidiate_list_asc)
        # letters = ["(A) ", "(B) ", "(C) ", "(D) "]
        letters = ["A", "B", "C", "D"]
        phrases = [place, option_1_tup[0], option_2_tup[0], option_3_tup[0]]
        logprobs = {place: gt_logprob, option_1_tup[0]: option_1_tup[1], option_2_tup[0]: option_2_tup[1], option_3_tup[0]: option_3_tup[1]}

        random.seed(option_seed)
        random.shuffle(phrases)
        MCQ_ans = letters[phrases.index(place)]
        options = dict(zip(letters, phrases))

        # options = []
        # for i in range(len(letters)):
        #     options.append(letters[i] + phrases[i])
        
        local_dict = {
            "id": gt_index,
            "context": {"subject": subject, "action": action},
            "target": {"place": place},
            "image": f"{gt_index:0>5}_{_subject}_{_action}_{_place}.{image_fmt}",
            "MCQ_options": options,
            "MCQ_ans": MCQ_ans,
            "logprobs": logprobs
        }
        return local_dict
    
    elif category == "action":
        subject, action, place = gt_triplet[0], gt_triplet[1], gt_triplet[2]
        _subject, _action, _place = underscore(subject), underscore(action), underscore(place)
        option_1_tup, option_2_tup, option_3_tup = select_items_from_list(candidiate_list_asc)
        # letters = ["(A) ", "(B) ", "(C) ", "(D) "]
        letters = ["A", "B", "C", "D"]
        phrases = [action, option_1_tup[0], option_2_tup[0], option_3_tup[0]]
        logprobs = {action: gt_logprob, option_1_tup[0]: option_1_tup[1], option_2_tup[0]: option_2_tup[1], option_3_tup[0]: option_3_tup[1]}

        random.seed(option_seed)
        random.shuffle(phrases)
        MCQ_ans = letters[phrases.index(action)]
        options = dict(zip(letters, phrases))

        # options = []
        # for i in range(len(letters)):
        #     options.append(letters[i] + phrases[i])
        
        local_dict = {
            "id": gt_index,
            "context": {"subject": subject, "place": place},
            "target": {"action": action},
            "image": f"{gt_index:0>5}_{_subject}_{_place}_{_action}.{image_fmt}",
            "MCQ_options": options,
            "MCQ_ans": MCQ_ans,
            "logprobs": logprobs
        }
    return local_dict

def create_input_list(cleaned_data_dict, scored_data, input_type):
    if input_type == 'place':
        ret = []
        idx = 1
        for subject in cleaned_data_dict:
            for action in cleaned_data_dict[subject]:
                for place_list in cleaned_data_dict[subject][action]:
                    gt_triplet = [subject, action, place_list[0]]
                    gt_logprob = place_list[1]
                    candidate_lst_asc = find_candidate_list_asc(complete_data_frame=scored_data, category="place", subject=subject, action=action, place=place_list[0])
                    input_dict = generate_text_inputs(gt_index=idx, gt_triplet=gt_triplet, gt_logprob=gt_logprob, candidiate_list_asc=candidate_lst_asc, category="place", option_seed=idx+3)
                    ret.append(input_dict)
                    idx += 1
        return ret

    elif input_type == 'action':
        ret = []
        idx = 1
        for subject in cleaned_data_dict:
            for place in cleaned_data_dict[subject]:
                for action_list in cleaned_data_dict[subject][place]:
                    gt_triplet = [subject, action_list[0], place]
                    gt_logprob = action_list[1]
                    candidate_lst_asc = find_candidate_list_asc(complete_data_frame=scored_data, category="action", subject=subject, action=action_list[0], place=place)
                    input_dict = generate_text_inputs(gt_index=idx, gt_triplet=gt_triplet, gt_logprob=gt_logprob, candidiate_list_asc=candidate_lst_asc, category="action", option_seed=idx+3)
                    ret.append(input_dict)
                    idx += 1
        return ret

if __name__ == '__main__':
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory) 
    place_gt_file_path = "subject_T20action_B20place_cleaned.json"
    action_gt_file_path = "subject_T20place_B20action_cleaned.json"
    scored_context_target_path = "context_target_exprs_K20_scored_vicuna_v15_13b.json"

    place_gt_dict = read_json(place_gt_file_path)
    action_gt_dict = read_json(action_gt_file_path)
    scored_context_target = read_json(scored_context_target_path)

    place_text_info = create_input_list(cleaned_data_dict=place_gt_dict, scored_data=scored_context_target, input_type="place")
    action_text_info = create_input_list(cleaned_data_dict=action_gt_dict, scored_data=scored_context_target, input_type="action")

    place_output_path = "Tplace_inputs.json"
    action_output_path = "Taction_inputs.json"

    write_json(place_output_path, place_text_info)
    write_json(action_output_path, action_text_info)