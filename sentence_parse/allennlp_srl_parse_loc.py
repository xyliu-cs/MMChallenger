from allennlp.predictors.predictor import Predictor
import json

# def extract_srl_information(predictor, sentence):
#     result = predictor.predict(sentence)
#     for verb in result['verbs']:
#         who, verb_phrase, at_where = None, None, None
#         to_whom = ""
#         for tag, word in zip(verb['tags'], result['words']):
#             if tag.startswith('B-ARG0'):
#                 who = word
#             elif tag.startswith('I-ARG0'):
#                 who += " " + word
#             elif tag == 'B-V':
#                 verb_phrase = word
#             elif tag.startswith('B-ARG1') or tag.startswith('I-ARG1'):
#                 to_whom += (" " if to_whom else "") + word
#             elif tag.startswith('B-ARGM-LOC'):
#                 at_where = word
#             elif tag.startswith('I-ARGM-LOC'):
#                 at_where += " " + word

#         # Combine verb and ARG1 to form a transitive verb phrase
#         verb_phrase = (verb_phrase or "") + (" " + to_whom if to_whom else "")
        
#         print(f"Sentence: {sentence}")
#         print(f"Who: {who}")
#         print(f"Verb Phrase: {verb_phrase}")
#         print(f"At Where: {at_where}\n")

def add_to_set(case_insensitive_set, items):
    for item in items:
        lower_item = item.lower()
        if lower_item not in case_insensitive_set:
            case_insensitive_set[lower_item] = item


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
            


if __name__ == '__main__':
    print("Good start")
    file_path = 'flores-200.json'
    print(f"Parsing corpus from {file_path}")

    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            sentences.append(data['sentence'])
    small_batch = sentences[:30]

    # must be placed at the same directory, using the relative path
    predictor = Predictor.from_path("./bert-base-srl-2020.11.19.tar.gz")

    i = 0
    where = {}
    for sentence in sentences:
        locs = extract_location_information(predictor, sentence)
        i += 1
        if locs:
            add_to_set(where, locs)
        if (i % 100 == 0):
            print(f"Processed {i} sentences")
    print(f"In total processed {i} sentences")

    locations = list(where.values())
    with open("locs.txt", "w") as file:
        for loc in locations:
            file.write(loc + '\n')
    print("Locations writing complete.")