from sentence_transformers import SentenceTransformer
import torch
from torch import nn
import json
import random



# determine the minimum moving items
def determine_moving_items(conflict_names, dt_num):
    trimmed_set = set()
    if dt_num == 0: # keep all regarding gt
        for conflict in conflict_names:
            if conflict[0] == 'ground_truth' or conflict[1] == 'ground_truth':
                if conflict[0] == 'ground_truth':
                    trimmed_set.add(conflict[1])
                else:
                    trimmed_set.add(conflict[0])
    else:
        for conflict in conflict_names:
            if conflict[0] == 'ground_truth' or conflict[1] == 'ground_truth':
                if conflict[0] == 'ground_truth':
                    trimmed_set.add(conflict[1])
                else:
                    trimmed_set.add(conflict[0])
        for conflict in conflict_names:
            if ('distractor' in conflict[0]) and ('distractor' in conflict[1]):
                if (conflict[0] not in trimmed_set) and (conflict[1] not in trimmed_set):
                    trimmed_set.add(conflict[1])
        for conflict in conflict_names:
            if ('distractor' in conflict[0]) and ('option' in conflict[1]):
                if conflict[0] not in trimmed_set: # if yes, don't need to add
                    trimmed_set.add(conflict[1])
            elif ('option' in conflict[0]) and ('distractor' in conflict[1]):
                if conflict[1] not in trimmed_set:
                    trimmed_set.add(conflict[0])
    return list(trimmed_set)



def check_phrase_closeness(check_model, phrase_dict, threshold, dt_num):
    raw_phrases = list(phrase_dict.values())
    phrase_tags = list(phrase_dict.keys())
    phrase_embs = check_model.encode(raw_phrases)

    all_paired_indices = [(x, y) for x in range(len(raw_phrases)) for y in range(x+1, len(raw_phrases))]
    all_paired_phrase_embs = [(phrase_embs[tup[0]], phrase_embs[tup[1]]) for tup in all_paired_indices]

    cos_sim = nn.CosineSimilarity(dim=0)
    all_paired_phrase_res = [cos_sim(torch.tensor(tup[0]), torch.tensor(tup[1])) for tup in all_paired_phrase_embs]

    conflict_tags = []
    for i in range(len(all_paired_phrase_res)):
        if all_paired_phrase_res[i] > threshold:
            conflict_tags.append((phrase_tags[all_paired_indices[i][0]], phrase_tags[all_paired_indices[i][1]]))
            print(f"Detected Closeness: {phrase_tags[all_paired_indices[i][0]]} [{raw_phrases[all_paired_indices[i][0]]}] - {phrase_tags[all_paired_indices[i][1]]} [{raw_phrases[all_paired_indices[i][1]]}] with {all_paired_phrase_res[i]} > threshold {threshold}")

    return determine_moving_items(conflict_tags, dt_num)



def construct_by_templates(candidate_dict, type):
    constructed_questions = []
    if type == 'verb':
        for group_id, sent_list in candidate_dict.items():
            local_verb_question = {}
            logprob_vp = {}
            vp_shortlist = []
            gt_subj, gt_be, gt_verb, gt_loc = None, None, None, None
            dt_verbs = {}
            has_distractor = False
            for sent in sent_list:
                if sent['tag'] == 'ground_truth':
                    gt_subj = sent['subj']
                    gt_be = sent['be']
                    gt_verb = sent['vp']
                    gt_loc = sent['loc']
                elif 'distractor' in sent['tag']: # distractor1, distractor2, ...
                    has_distractor = True
                    seq = sent['tag'].removeprefix("distractor")
                    dt_verbs[f"dt_vp{seq}"] = sent['vp']
                
                logprob_vp[sent['vp']] = sent['logprob'] # it represents the sentence's ppl
                vp_shortlist.append(sent['vp'])

            assert(any([gt_subj, gt_be, gt_verb, gt_loc]) != None)
            option_num = len(vp_shortlist)

            local_verb_question['id'] = group_id
            local_verb_question['subj'] = gt_subj
            local_verb_question['be'] = gt_be
            local_verb_question['loc'] = gt_loc
            local_verb_question['gt_vp'] = gt_verb
            for dt_key in dt_verbs:
                local_verb_question[dt_key] = dt_verbs[dt_key]
            local_verb_question['logprob'] = logprob_vp

            random.shuffle(vp_shortlist)  # we shuffle the option's list
            choice_ans_idx = vp_shortlist.index(gt_verb)
            choice_dt_idx = {}
            for dt_key in dt_verbs:
                choice_dt_idx[dt_key] = vp_shortlist.index(dt_verbs[dt_key])


            # Multiple Choice Questions, add here for more templates
            choice_question = f"Question: What {gt_be} {gt_subj} doing {gt_loc}?"

            choice_option_list = ['Options:']
            letters = [chr(i) for i in range(ord('A'), ord('A') + option_num)]
            for i in range(option_num):

                choice_option_list.append(f'({letters[i]}) {vp_shortlist[i]}.')
            choice_options = ' '.join(choice_option_list)

            choice_postfix_instructblip = "Answer:"
            choice_postfix_llava = "Answer with the option's letter from the given choices directly."

            local_verb_question[f'choice instructblip'] = choice_question + ' ' + choice_options + ' ' + choice_postfix_instructblip
            local_verb_question[f'choice llava'] = choice_question + '\n' + choice_options + '\n' + choice_postfix_llava
          
            local_verb_question['choice answer'] = letters[choice_ans_idx]
            for dt_key in choice_dt_idx:
                seq = dt_key.removeprefix("dt_vp")
                local_verb_question[f'distractor{seq} answer'] = letters[choice_dt_idx[dt_key]]


            # Binary questions, add here for more templates
            local_verb_question['binary-yes'] = f"Does the image show that {gt_subj} {gt_be} {gt_verb} {gt_loc}?"
            local_verb_question['binary-yes answer'] = "Yes"
            
            if has_distractor:
                local_verb_question['binary-no'] = f"Does the image show that {gt_subj} {gt_be} {dt_verbs['dt_vp1']} {gt_loc}?" # Always choose the top distractor 
                local_verb_question['binary-no answer'] = "No"
            
            min_logprob_verb = min(logprob_vp, key=logprob_vp.get)
            local_verb_question['binary-cp'] = f"Does the image show that {gt_subj} {gt_be} {min_logprob_verb} {gt_loc}?"
            local_verb_question['binary-cp answer'] = "No"


            # Open questions, add here for more templates
            local_verb_question['open llava'] = f"What {gt_be} {gt_subj} doing {gt_loc}? Answer the question using a single word or phrase."
            local_verb_question['open instructblip'] = f"What {gt_be} {gt_subj} doing {gt_loc}? Short Answer:"

            local_verb_question['open answer keyword'] = gt_verb
            local_verb_question['text'] = f"{gt_subj} {gt_be} {gt_verb} {gt_loc}."

            constructed_questions.append(local_verb_question)

    elif type == 'location':
        for group_id, sent_list in candidate_dict.items():
            local_loc_question = {}
            logprob_loc = {}
            loc_shortlist = []
            gt_subj, gt_be, gt_verb, gt_loc = None, None, None, None
            dt_locs = {}
            has_distractor = False
            for sent in sent_list:
                if sent['tag'] == 'ground_truth':
                    gt_subj = sent['subj']
                    gt_be = sent['be']
                    gt_verb = sent['vp']
                    gt_loc = sent['loc']
                elif 'distractor' in sent['tag']: # distractor1, distractor2, ...
                    has_distractor = True
                    seq = sent['tag'].removeprefix("distractor")
                    dt_locs[f"dt_loc{seq}"] = sent['loc']
                
                logprob_loc[sent['loc']] = sent['logprob'] # it represents the sentence's ppl
                loc_shortlist.append(sent['loc'])

            assert(any([gt_subj, gt_be, gt_verb, gt_loc]) != None)
            option_num = len(loc_shortlist)

            local_loc_question['id'] = group_id
            local_loc_question['subj'] = gt_subj
            local_loc_question['be'] = gt_be
            local_loc_question['vp'] = gt_verb
            local_loc_question['gt_loc'] = gt_loc
            for dt_key in dt_locs:
                local_loc_question[dt_key] = dt_locs[dt_key]
            local_loc_question['logprob'] = logprob_loc

            random.shuffle(loc_shortlist)  # we shuffle the option's list
            choice_ans_idx = loc_shortlist.index(gt_loc)
            choice_dt_idx = {}
            for dt_key in dt_locs:
                choice_dt_idx[dt_key] = loc_shortlist.index(dt_locs[dt_key])


            # Multiple Choice Questions, add here for more templates
            choice_question = f"Question: Where {gt_be} {gt_subj} {gt_verb}?"

            choice_option_list = ['Options:']
            letters = [chr(i) for i in range(ord('A'), ord('A') + option_num)]
            for i in range(option_num):
                choice_option_list.append(f'({letters[i]}) {loc_shortlist[i]}.')
            choice_options = ' '.join(choice_option_list)

            choice_postfix_instructblip = "Answer:"
            choice_postfix_llava = "Answer with the option's letter from the given choices directly."

            local_loc_question[f'choice instructblip'] = choice_question + ' ' + choice_options + ' ' + choice_postfix_instructblip
            local_loc_question[f'choice llava'] = choice_question + '\n' + choice_options + '\n' + choice_postfix_llava

            for dt_key in choice_dt_idx:
                seq = dt_key.removeprefix("dt_loc")
                local_loc_question[f'distractor{seq} answer'] = letters[choice_dt_idx[dt_key]]


            # Binary questions, add here for more templates
            local_loc_question['binary-yes'] = f"Does the image show that {gt_subj} {gt_be} {gt_verb} {gt_loc}?"
            local_loc_question['binary-yes answer'] = "Yes"
            
            if has_distractor:
                local_loc_question['binary-no'] = f"Does the image show that {gt_subj} {gt_be} {gt_verb} {dt_locs['dt_loc1']}?" # Always choose the top distractor 
                local_loc_question['binary-no answer'] = "No"
            
            min_logprob_loc = min(logprob_loc, key=logprob_loc.get)
            local_loc_question['binary-cp'] = f"Does the image show that {gt_subj} {gt_be} {gt_verb} {min_logprob_loc}?"
            local_loc_question['binary-cp answer'] = "No"


            # Open questions, add here for more templates
            local_loc_question['open llava'] = f"Where {gt_be} {gt_subj} {gt_verb}? Answer the question using a single word or phrase."
            local_loc_question['open instructblip'] = f"Where {gt_be} {gt_subj} {gt_verb}? Short Answer:"

            local_loc_question['open answer keyword'] = gt_loc
            local_loc_question['text'] = f"{gt_subj} {gt_be} {gt_verb} {gt_loc}."

            constructed_questions.append(local_loc_question)
    else:
        raise ValueError (f"Unsupported question template type {type}")
    
    return constructed_questions



def clean_generate(sentence_ppl_path, base_path, output_file, type, check_model, pre_method, option_num=5, num_distractors=1, ban_list={'verb':[], 'loc': []}, verbose=True, dump_intermediate=False):
    assert(option_num >= num_distractors + 1)
    # assert(num_distractors <= 2)
    
    with open(sentence_ppl_path, 'r') as f:
        sentence_data = json.load(f)
    
    with open(base_path, 'r') as f:
        base_sentences = json.load(f)
    
    def print_v(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    print_v(f"\n[Program Info] Banned verb phrases: {ban_list['verb']}")
    print_v(f"[Program Info] Banned location phrases: {ban_list['loc']}")
    
    # universal settings
    gt_start_index = -10
    non_dt_options = option_num - 1 - num_distractors
    candidate_groups = {}
    # verb phrase test case construction
    if type == 'verb':
        ctr = 1
        for sent_dict in base_sentences: # base sentence itself can be a distractor
            subject = sent_dict["subj"]
            verb_phrase = sent_dict["vp"]
            location = sent_dict["loc"]
            control_batch = []
            for sent in sentence_data:
                if (sent['subj'] == subject) and (sent['loc'] == location) and (sent['vp'] not in ban_list['verb']):
                    control_batch.append(sent)
            sorted_batch = sorted(control_batch, key=lambda x: x['logprob'], reverse=True)

            options = {}
            options_idx = {}
            
            # put ground truth
            options_idx["ground_truth"] = gt_start_index
            options['ground_truth'] =  sorted_batch[gt_start_index].copy()
            # put options
            for i in range(non_dt_options):
                options_idx[f"option{i+1}"] = -(i+1) 
                options[f"option{i+1}"] = sorted_batch[-(i+1)].copy()
            # put distractors
            if num_distractors >= 1:
                bs_idx = next((i for i, d in enumerate(sorted_batch) if d.get('vp') == verb_phrase), -1)
                if bs_idx != 0 and pre_method == 'default':
                    raise ValueError(f"The first distractor should be have a local minimum ppl by {pre_method}, yet it is placed at {bs_idx}th place")
                else:
                    options_idx["distractor1"] = bs_idx
                    options["distractor1"] = sorted_batch[bs_idx].copy()
                for j in range(num_distractors-1):
                    options_idx[f"distractor{j+2}"] = bs_idx + j + 1
                    options[f"distractor{j+2}"] = sorted_batch[bs_idx+j+1].copy()

            need_further_iteration = True
            while need_further_iteration:
                phrase_dict = {tag: options[tag]['vp'] for tag in options}
                print_v(f"Group {ctr}: {phrase_dict.values()}")
                tags_need_to_move = check_phrase_closeness(check_model, phrase_dict, 0.78, num_distractors) # verb threshold

                if len(tags_need_to_move) == 0:
                    print_v(f"Group {ctr} Succeed.\n========")
                    for tag, sent_dict in options.items():
                        sent_dict['tag'] = tag
                        sent_dict['gid'] = ctr
                    candidate_groups[ctr] = list(options.values())
                    ctr += 1
                    need_further_iteration = False  # normal exit of the while loop
                
                else:
                    for tag in tags_need_to_move:
                        print_v(f"Moving {tag} [{options[tag]['vp']}]")
                        if 'distractor' in tag:
                            options_idx[tag] += num_distractors
                            options[tag] = sorted_batch[options_idx[tag]].copy()
                        elif 'option' in tag:
                            options_idx[tag] -= non_dt_options
                            if options_idx[tag] <= gt_start_index:
                                print_v(f"[Program Info] Give up this group due to {tag} passing ground-truth index {gt_start_index}")
                                print_v(f"[Program Info] Last components {phrase_dict.values()}")
                                need_further_iteration = False
                                break
                            options[tag] = sorted_batch[options_idx[tag]].copy()
    
    elif type == 'location':
        ctr = 1
        for sent_dict in base_sentences: # base sentence itself can be a distractor
            subject = sent_dict["subj"]
            verb_phrase = sent_dict["vp"]
            location = sent_dict["loc"]
            control_batch = []
            for sent in sentence_data:
                if (sent['subj'] == subject) and (sent['vp'] == verb_phrase) and (sent['loc'] not in ban_list['loc']):
                    control_batch.append(sent)
            sorted_batch = sorted(control_batch, key=lambda x: x['logprob'], reverse=True)

            options = {}
            options_idx = {}
            # put ground truth
            options_idx["ground_truth"] = gt_start_index
            options['ground_truth'] =  sorted_batch[gt_start_index].copy()
            # put options
            for i in range(non_dt_options):
                options_idx[f"option{i+1}"] = -(i+1) 
                options[f"option{i+1}"] = sorted_batch[-(i+1)].copy()
            # put distractors
            if num_distractors >= 1:
                bs_idx = next((i for i, d in enumerate(sorted_batch) if d.get('loc') == location), -1)
                if bs_idx != 0 and pre_method == 'default':
                    raise ValueError(f"The first distractor should be have a local minimum ppl by {pre_method}, yet it is placed at {bs_idx}th place")
                else:
                    options_idx["distractor1"] = bs_idx
                    options["distractor1"] = sorted_batch[bs_idx].copy()
                for j in range(num_distractors-1):
                    options_idx[f"distractor{j+2}"] = bs_idx + j + 1
                    options[f"distractor{j+2}"] = sorted_batch[bs_idx+j+1].copy()

            need_further_iteration = True
            while need_further_iteration:
                phrase_dict = {tag: options[tag]['loc'] for tag in options}
                print_v(f"Group {ctr}: {phrase_dict.values()}")
                tags_need_to_move = check_phrase_closeness(check_model, phrase_dict, 0.7, num_distractors) # location threshold

                if len(tags_need_to_move) == 0:
                    print_v(f"Group {ctr} Succeed.\n========")
                    for tag, sent_dict in options.items():
                        sent_dict['tag'] = tag
                        sent_dict['gid'] = ctr
                    candidate_groups[ctr] = list(options.values())
                    ctr += 1
                    need_further_iteration = False  # normal exit of the while loop
                
                else:
                    for tag in tags_need_to_move:
                        print_v(f"Moving {tag} [{options[tag]['loc']}]")
                        if 'distractor' in tag:
                            options_idx[tag] += num_distractors  # step size
                            options[tag] = sorted_batch[options_idx[tag]].copy()
                        elif 'option' in tag:
                            options_idx[tag] -= non_dt_options
                            if options_idx[tag] <= gt_start_index:
                                print_v(f"[Program Info] Give up this group due to {tag} passing ground-truth index {gt_start_index}")
                                print_v(f"[Program Info] Last components {phrase_dict.values()}")
                                need_further_iteration = False
                                break
                            options[tag] = sorted_batch[options_idx[tag]].copy()
    
    else:
        raise ValueError (f"Unsupported question type {type}")
    
    print_v("="*40)      
    print_v(f"Total valid question grounps ({type}): {ctr-1}")
    if dump_intermediate:
        last_slash_idx = output_file.rfind('/')
        if last_slash_idx == -1:
            path = f"./{type}_candidates_dump.json"
        else:
            path = output_file[:last_slash_idx+1] + f"{type}_candidates_dump.json"
        print_v(f"\n[Program Info] Dumping intermediates groups to {path}")
        with open(path, 'w') as f:
            json.dump(candidate_groups, f, indent=4)
        
    question_items = construct_by_templates(candidate_groups, type)
    
    print_v(f"\n[Program Info] Writing question file to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(question_items, f, indent=4)




if __name__ == '__main__':
    model = SentenceTransformer('whaleloops/phrase-bert')
    sentence_ppl = "/home/liu/test_resources/sentence_lookup/sents_logprob_0328.json"
    base_sentences = "/home/liu/test_resources/base_sentences/base_sentences_0329.json"
    outfile = '/home/liu/test_resources/input_questions/location_questions.json'
    ban_words = {'verb': [], 'loc':[]}
    clean_generate(sentence_ppl, base_sentences, outfile, 'location', model, 'reservoir', option_num=4, num_distractors=1, ban_list=ban_words, dump_intermediate=False)