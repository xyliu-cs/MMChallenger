import json, os
from collections import defaultdict


# cartesian product
def build_tuples(data_dict_1: dict, category_1: str, data_dict_2: dict, category_2: str) -> list:
    ret = []
    id = 0
    for phrase_i in data_dict_1:
        for phrase_j in data_dict_2:
            ret.append({"expr_id": id, "expr": f"{phrase_i} {phrase_j}", category_1: phrase_i, category_2: phrase_j})
            id += 1
    return ret


if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory) 


    subject_path = "../input_data/Concept_Sets/after_manual/OMCS-SUB-100k-T1000-cleaned-100.json"
    action_path = "../input_data/Concept_Sets/after_manual/OMCS-ACT-100k-T1000-cleaned-150-VBG.json"
    place_path = "../input_data/Concept_Sets/after_manual/OMCS-PLA-100k-T1000-cleaned-150.json"

    output_path = "context_exprs.json"    

    with open(subject_path, 'r') as f:
        subj_data = json.load(f)
    with open(action_path, 'r') as f:
        action_data = json.load(f)
    with open(place_path, 'r') as f:
        place_data = json.load(f)
    
    # subj - action
    subj_act_tuples = build_tuples(subj_data, "subj", action_data, "action")
    assert len(subj_act_tuples) == len(subj_data) * len(action_data), "subj x act length does not match"
    subj_pla_tuples = build_tuples(subj_data, "subj", place_data, "place")
    assert len(subj_pla_tuples) == len(subj_data) * len(place_data), "subj x pla length does not match"
    total = subj_act_tuples + subj_pla_tuples

    with open(output_path, 'w') as f:
        json.dump(total, f)
    
    print(f"Write {len(total)} context expressions to {output_path}")

    # score the output context_exprs.json and place it to context_exprs_scored.json
    # need to integrate from logprob here


    input_file = "context_exprs_scored.json"
    # group by subject
    actions_groups_by_subjects = defaultdict(list)
    places_groups_by_subjects = defaultdict(list)

    K = 20

    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for index, expr_dict in enumerate(data):
        # subject - action: 0 - 15000
        if index == expr_dict["expr_id"]:
            actions_groups_by_subjects[expr_dict["subj"]].append((expr_dict["action"], expr_dict["logprob"]))
        # subject - action: 0 - 15000
        else:
            places_groups_by_subjects[expr_dict["subj"]].append((expr_dict["place"], expr_dict["logprob"]))

    # Sorting each list in actions_groups_by_subjects by logprob
    for subject in actions_groups_by_subjects:
        actions_groups_by_subjects[subject] = sorted(actions_groups_by_subjects[subject], key=lambda x: x[1], reverse=True)[:K]

    # Sorting each list in places_groups_by_subjects by logprob
    for subject in places_groups_by_subjects:
        places_groups_by_subjects[subject] = sorted(places_groups_by_subjects[subject], key=lambda x: x[1], reverse=True)[:K]
    
    output_f1 = f"SUB-ACT_K{K}.json"
    output_f2 = f"SUB-PLA_K{K}.json"


    with open(output_f1, 'w') as f:
        json.dump(actions_groups_by_subjects, f, indent=2)
    print(f"Write {len(actions_groups_by_subjects) * K} context tuples with top-{K} log-prob to {output_f1}")

    with open(output_f2, 'w') as f:
        json.dump(places_groups_by_subjects, f, indent=2)
    print(f"Write {len(places_groups_by_subjects) * K} context tuples with top-{K} log-prob to {output_f2}")

    # construct context-target triplets
    action_set = "../input_data/Concept_Sets/after_manual/OMCS-ACT-100k-T1000-cleaned-150-VBG.json"
    place_set = "../input_data/Concept_Sets/after_manual/OMCS-PLA-100k-T1000-cleaned-150.json"

    with open(action_set, 'r') as f:
        action_data = json.load(f)
    action_phrases = action_data.keys()
    with open(place_set, 'r') as f:
        place_data= json.load(f)
    place_phrases = place_data.keys()


    place_to_subject_action = []
    action_to_subject_place = []

    for subject, tup_list in actions_groups_by_subjects.items():
        for action_tup in tup_list:
            action = action_tup[0]
            logprob = action_tup[1]
            for place in place_phrases:
                local_dict = {"category": 0, "expr": f"{subject} {action} {place}", "subject": subject, "action": action, "place": place}
                place_to_subject_action.append(local_dict)
    assert len(place_to_subject_action) == len(actions_groups_by_subjects) * K * len(place_data), "subject x K (actions) x places length does not match"
    
    for subject, tup_list in places_groups_by_subjects.items():
        for place_tup in tup_list:
            place = place_tup[0]
            logprob = place_tup[1]
            for action in action_phrases:
                local_dict = {"category": 1, "expr": f"{subject} {place} {action}", "subject": subject, "action": action, "place": place}
                action_to_subject_place.append(local_dict)
    assert len(action_to_subject_place) == len(places_groups_by_subjects) * K * len(action_data), "subject x K (places) x actions length does not match"

    total = place_to_subject_action + action_to_subject_place

    output_f3 = f"context_target_exprs_K{K}.json"

    with open(output_f3, 'w') as f:
        json.dump(total, f)
    print(f"Write {len(total)} context-target triplets to {output_f3}")