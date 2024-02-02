from allennlp.predictors.predictor import Predictor
from collections import Counter
import argparse

parser = argparse.ArgumentParser(description='Preprocess the input files.')
parser.add_argument('input_file_path', type=str, help='The path to the input file (raw sentences file)')
parser.add_argument('output_file_path', type=str, help='The path to the output file (location word phrase list)')

# Execute the parse_args() method
args = parser.parse_args()
input_file_path = args.input_file_path
output_file_path = args.output_file_path
predictor_path = "./bert-base-srl-2020.11.19.tar.gz" # need to be at the same directory


def add_to_set(case_insensitive_set, items):
    for item in items:
        lower_item = item.lower()
        case_insensitive_set[lower_item] = case_insensitive_set.get(lower_item, 0) + 1


def get_top_n_items(dictionary, n=150):
    counter = Counter(dictionary)
    return counter.most_common(n)


def extract_location_information(predictor, sentence):
    result = predictor.predict(sentence)
    locations = []
    current_location = None
    for verb in result['verbs']:
        for tag, word in zip(verb['tags'], result['words']):
            if tag.startswith('B-ARGM-LOC') or tag.startswith('B-ARGM-DIR'):
                if current_location:
                    locations.append(current_location)
                current_location = word
            elif tag.startswith('I-ARGM-LOC') or tag.startswith('I-ARGM-DIR'):
                current_location += " " + word

        if current_location:
            locations.append(current_location)
            current_location = None

    if locations:
        return locations
            

print(f"Loading predictor from {predictor_path} ...")
predictor = Predictor.from_path(predictor_path)

with open(input_file_path, 'r', encoding='utf-8') as file:
    sentences = [line.strip() for line in file.readlines()]

sentence_batch = sentences[:20000] # parse the first 20000 for pilot research
print(f"Parsing {len(sentence_batch)} sentences from {input_file_path} ...")

i = 0
where = {}
for sentence in sentence_batch:
    locs = extract_location_information(predictor, sentence)
    i += 1
    if locs:
        add_to_set(where, locs)
    if (i == 10):
        print(f"Started parsing, already took 10, the progress will be reported at every thousand.")
    if (i % 1000 == 0):
        print(f"Processed {i} sentences")
print(f"Finished parsing, in total processed {i} sentences")

top_verb_phrases = get_top_n_items(where, 500) # keep the top 500 for manual filter
print(f"Keeping the top 500 frequent words, writing to {output_file_path} ...")
with open(output_file_path, "w") as file:
    for loc, freq in top_verb_phrases:
        file.write(f"{loc}: {freq}\n")
print("Finished writing locations, program exiting.")