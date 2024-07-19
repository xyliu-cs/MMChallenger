import json
from lemminflect import getInflection, getLemma

TOP_FREQUENCY_PHRASES = 1000
PARTICLES = ["a", "the"]

def remove_similar_phrases(data):
    cleaned_data = {}
    for phrase, count in data.items():
        words = phrase.split()
        if words[0] in PARTICLES:
            key = ' '.join(words[1:])
            if key in cleaned_data:
                if count > cleaned_data[key][1]:
                    cleaned_data[key] = (phrase, count)
            else:
                cleaned_data[key] = (phrase, count)
        else:
            cleaned_data[phrase] = (phrase, count)
    
    # Rebuild the dictionary with the original phrases
    result = {v[0]: v[1] for v in cleaned_data.values()}
    return result


def get_present_continuous(verb):
    lemma = getLemma(verb, upos='VERB')
    if lemma:
        v_ing = getInflection(lemma[0], tag='VBG')
        if v_ing:
            return v_ing[0]
        else:
            print(f"No VBG from verb {lemma}")
            return verb
    else:
        print(f"No lemma from verb {verb}")
        return verb


if __name__ == "__main__":
    with open('../input_data/OMCS/split_100k/OMCS-SUB-100k.json', 'r') as f:
        subject_all = json.load(f)
    with open('../input_data/OMCS/split_100k/OMCS-ACT-100k.json', 'r') as f:
        action_all  = json.load(f)
    with open('../input_data/OMCS/split_100k/OMCS-PLA-100k.json', 'r') as f:
        place_all = json.load(f)

    print(f"Total subjects: {len(subject_all)}")
    print(f"Total actions: {len(action_all)}")
    print(f"Total places: {len(place_all)}")

    print(f"Filtering the top {TOP_FREQUENCY_PHRASES} ones for each category based on phrase frequencies")
    targets= [subject_all, action_all, place_all]
    names = ["SUB", "ACT", "PLA"]
    tops = []
    for idx in range(len(targets)):
        target_data = targets[idx]
        target_data_srt = sorted(target_data.items(), key=lambda item: item[1], reverse=True)
        target_data_keep = target_data_srt[:TOP_FREQUENCY_PHRASES]
        target_data_dict = {k: v for k, v in target_data_keep}
        tops.append(target_data_dict)
        fname = f"OMCS-{names[idx]}-100k-T{TOP_FREQUENCY_PHRASES}.json"
        with open(f"../input_data/Concept_Sets/before_manual/{fname}", "w") as f:
            json.dump(target_data_dict, f, indent=4)
        fname_2 = f"OMCS-{names[idx]}-100k-T{TOP_FREQUENCY_PHRASES}-cleaned.json"
        if names[idx] == "SUB":
            target_data_dict = remove_similar_phrases(target_data_dict)
        with open(f"../input_data/Concept_Sets/after_manual/{fname_2}", "w") as f:
            json.dump(target_data_dict, f, indent=4)

    
    # unify verb tense
    # input_file_path = '../input_data/Concept_Sets/after_manual/OMCS-ACT-100k-T1000-cleaned-150.json'
    # output_file_path = '../input_data/Concept_Sets/after_manual/OMCS-ACT-100k-T1000-cleaned-150-VBG.json'

    # with open(input_file_path, 'r') as f:
    #     data = json.load(f)

    # vbg_dict = {}
    # for phrase in data:
    #     words = phrase.split()
    #     verb_ing = get_present_continuous(words[0])
    #     words[0] = verb_ing
    #     gerund = ' '.join(words)
    #     vbg_dict[gerund] = data[phrase] 

    # vbg_duplicates_removed = {}
    # for phrase, frequency in vbg_dict.items():
    #     # remove duplicates after mannual inspection
    #     if phrase in vbg_duplicates_removed:
    #         if vbg_duplicates_removed[phrase] < frequency:
    #             vbg_duplicates_removed[phrase] = frequency
    #     else:
    #         vbg_duplicates_removed[phrase] = frequency

    # with open(output_file_path, 'w') as f:
    #     json.dump(vbg_duplicates_removed, f, indent=4)