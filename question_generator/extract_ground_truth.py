import json

def extract_ground_truth(input_path, output_path, verbose=True):
    def print_v(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    with open(input_path, 'r') as f:
        question_list = json.load(f)
    
    type = None
    if 'gt_vp' in question_list[0].keys():
        type = 'verb'
        print_v(f"[Program Info] Indentified verb question, extracting ground-truth sentences")
    elif 'gt_loc' in question_list[0].keys():
        type = 'loc'
        print_v(f"[Program Info] Indentified location question, extracting ground-truth sentences")
    else:
        raise TypeError("Unsupported question file type")

    ground_truth = {}
    if type == 'verb':
        for question_dict in question_list:
            essentials = {}
            for key_name in ['subj', 'be', 'gt_vp', 'loc']:
                essentials[key_name] = question_dict[key_name]
            essentials['text'] = f"{essentials['subj']} {essentials['be']} {essentials['gt_vp']} {essentials['loc']}"
            print_v(essentials['text'])
            ground_truth[question_dict['id']] = essentials

    elif type == 'loc':
        for question_dict in question_list:
            essentials = {}
            for key_name in ['subj', 'be', 'vp', 'gt_loc']:
                essentials[key_name] = question_dict[key_name]
            essentials['text'] = f"{essentials['subj']} {essentials['be']} {essentials['vp']} {essentials['gt_loc']}"
            print_v(essentials['text'])
            ground_truth[question_dict['id']] = essentials

    print_v("="*15)
    print_v(f"Total Question - Ground truth: {len(ground_truth.keys())}")

    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=4)
    
    return ground_truth


if __name__ == '__main__':
    input_path = "/home/liu/test_resources/input_questions/verb_questions_vicuna_0302_partB.json"
    output_path = "/home/liu/test_resources/ground_truth/verb_0302_partB_ground_truth.json"

    extract_ground_truth(input_path, output_path)