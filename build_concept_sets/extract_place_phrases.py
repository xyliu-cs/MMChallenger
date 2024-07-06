from allennlp.predictors.predictor import Predictor
from tqdm import tqdm
import argparse, re, json

parser = argparse.ArgumentParser(description='Preprocess the input files.')
parser.add_argument('input_file_path', type=str, help='The path to the input file (raw sentences file)')
parser.add_argument('output_file_path', type=str, help='The path to the output file (location word phrase list)')

# Execute the parse_args() method
args = parser.parse_args()
input_file_path = args.input_file_path
output_file_path = args.output_file_path
USE_SPLIT = 100000


def construct_phrase_occurrence_dict(phrases_list):
    phrase_dict = {} 
    for phrase in phrases_list:
        phrase_lower = phrase.lower()
        if phrase_lower in phrase_dict:
            phrase_dict[phrase_lower] += 1
        else:
            phrase_dict[phrase_lower] = 1
    return phrase_dict


# Extract Place phrases from the input sentence
# 
# Input Sentence:
# "She found her keys hidden under the stack of books."
# 
# Example SRL results:
# {'verbs': [{'verb': 'found', 'description': '[ARG0: She] [V: found] [ARG1: her keys hidden under the stack of books] .', 
# 'tags': ['B-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O']}, 
# {'verb': 'hidden', 'description': 'She found [ARG1: her keys] [V: hidden] [ARGM-LOC: under the stack of books] .', 
# 'tags': ['O', 'O', 'B-ARG1', 'I-ARG1', 'B-V', 'B-ARGM-LOC', 'I-ARGM-LOC', 'I-ARGM-LOC', 'I-ARGM-LOC', 'I-ARGM-LOC', 'O']}], 
# 'words': ['She', 'found', 'her', 'keys', 'hidden', 'under', 'the', 'stack', 'of', 'books', '.']}
# 
def extract_place_phrases(srl_result, min_len=3, max_len=4):
    location_phrases = []
    for verb in srl_result['verbs']:
        description = verb['description']
        # Find all ARGM-LOC tagged phrases
        loc_matches = re.findall(r'\[ARGM-LOC: ([^\]]+)\]', description)
        # Filter for phrase length between 3 and 4 words
        for loc in loc_matches:
            word_count = len(loc.split())
            if min_len <= word_count <= max_len:
                location_phrases.append(loc)
    return location_phrases
    

if __name__ == "__main__":
    model_path = "./bert-base-srl-2020.11.19.tar.gz" # need to be at the same directory
    print(f"Loading predictor from {model_path} ...")
    predictor = Predictor.from_path(model_path)

    with open(input_file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file.readlines()]
    sentence_batch = sentences[:USE_SPLIT] # parse the first 100K for research
    print(f"Parsing {len(sentence_batch)} sentences from {input_file_path} ...")

    place_phrases = []
    for sentence in tqdm(sentence_batch, desc="Parsing Sentences"):
        srl_result = predictor.predict(sentence)
        places = extract_place_phrases(srl_result)
        if places:
            place_phrases.extend(places)
    
    place_occurence_dict = construct_phrase_occurrence_dict(place_phrases)

    with open(output_file_path, "w") as f:
        json.dump(place_phrases, f)
    print(f"Finished extracting action phrases, file written to {output_file_path}.")