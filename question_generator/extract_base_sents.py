import json
from collections import deque

def extract_base_sentences(input_path, output_path, ban_list, method='default', reservoir_capacity=5, verbose=True):
    def print_v(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    with open(input_path, 'r') as f:
        sentence_data = json.load(f)
    
    subjects = []
    for sent_dict in sentence_data:
        if sent_dict['subj'] not in subjects:
            subjects.append(sent_dict['subj'])
    
    print_v(f"[Program Info] Banned verb phrases: {ban_list['verb']}")
    print_v(f"[Program Info] Banned location phrases: {ban_list['loc']}")

    base_sentences = []
    # select sentence with global minimum perplexity as base
    if method == 'default': 
        print_v(f"[Program Info] Applying default method to select sentence with global minimum perplexity as base")
        for subject in subjects:
            compare_batch = []
            for sent_dict in sentence_data:
                if (sent_dict['subj'] == subject) and (sent_dict['vp'] not in ban_list['verb']) and (sent_dict['loc'] not in ban_list['loc']):
                    compare_batch.append(sent_dict)
            # assert(len(compare_batch) == 5000)
            sorted_compare_list = sorted(compare_batch, key=lambda x: x['ppl'])
            print_v(f"\n{subject}\n========")
            for i in sorted_compare_list[:5]:
                print_v(f"{i['text']}, ppl = {i['ppl']}")
            base_sentences.append(sorted_compare_list[0])

    
    # keep a fixed number (resevoir_samples) of same [verb phrase + location] in the base
    elif method == 'reservoir':
        print_v(f"[Program Info] Applying reservoirs for the [verb + location] combination with reservoir capacity = {reservoir_capacity}")
        subjects_queue = deque(subjects)
        subject_ppl_order = {}

        for subj in subjects:
            subject_ppl_order[subj] = 0

        reservoirs = {}
        while subjects_queue:
            subject = subjects_queue.popleft()
            ppl_index = subject_ppl_order[subject]
            
            compare_batch = []
            for sent_dict in sentence_data:
                if (sent_dict['subj'] == subject) and (sent_dict['vp'] not in ban_list['verb']) and (sent_dict['loc'] not in ban_list['loc']):
                    compare_batch.append(sent_dict)
                # assert(len(compare_batch) == 5000)
            sorted_compare_list = sorted(compare_batch, key=lambda x: x['ppl'])

            item = sorted_compare_list[ppl_index]
            verb_loc = item['vp'] + ' ' + item['loc']  # e.g. [riding a bicycle in the countryside]
            
            # If the 'text' value is not already a reservoir, initialize it first
            if verb_loc not in reservoirs:
                reservoirs[verb_loc] = []

            if len(reservoirs[verb_loc]) < reservoir_capacity:
                reservoirs[verb_loc].append(item)         
            else:
                # If the reservoir is full, pop the max perplexity sentence
                reservoirs[verb_loc].append(item)
                sorted_ppl_subject = sorted(reservoirs[verb_loc], key=lambda x: x['ppl'])
                replace_subj = sorted_ppl_subject[-1]['subj']
                subject_ppl_order[replace_subj] += 1
                subjects_queue.append(replace_subj)
                for i, d in enumerate(reservoirs[verb_loc]):
                    if d['subj'] == replace_subj:
                        print_v(f'[Program Info] Dequeuing [{d['text']}]')
                        del reservoirs[verb_loc][i]
                        break
                assert(len(reservoirs[verb_loc]) == reservoir_capacity)
            
        # flatten the reservoirs
        for reservoir in reservoirs.values():
            for sent_dict in reservoir:
                base_sentences.append(sent_dict)
        
        for base in base_sentences:
            print_v(f"\n[{base['subj']}]")
            print_v(f"{base['text']}")
 
    else:
        raise ValueError("Method should be passed as 'default' or 'reservoir'.")

    # branches merges here
    base_sentences_sorted_ids = sorted(base_sentences, key=lambda x: x['id'])
    with open(output_path, 'w') as f:
        json.dump(base_sentences_sorted_ids, f, indent=4)
    
    return base_sentences_sorted_ids



if __name__ == '__main__':
    input_path = "/home/liu/test_resources/sentence_lookup/sentence_lookup_vicuna_2024-03-14_auto.json"
    output_path = "/home/liu/test_resources/base_sentences/base_sentences_0316_reservoir.json"
    ban_words = {'verb': ['riding a bicycle'], 'loc':[]}
    extract_base_sentences(input_path, output_path, ban_words, method = 'reservoir')