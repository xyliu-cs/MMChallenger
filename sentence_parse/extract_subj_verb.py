import spacy
from spacy.matcher import Matcher
from collections import Counter
import json

def extract_action_subjects(doc):
    named_entities = set(ent.text for ent in doc.ents)  # Exclude named entities.
    action_subjects = []
    for chunk in doc.noun_chunks:
        # Filter based on dependency tags, exclude personal pronouns, and ensure it's an action verb.
        if chunk.root.dep_ in ["nsubj", "nsubjpass"] and chunk.root.pos_ != "PRON":
            # Ensure the verb linked to the subject is an action verb
            head_verb = chunk.root.head
            if head_verb.pos_ == "VERB" and not any(ne in chunk.text or chunk.text in ne for ne in named_entities):
                subject = chunk.text
                action_subjects.append(subject)
    return action_subjects


def extract_verbs_phrase(doc, nlp):
    verb_phrases = []
    matcher = Matcher(nlp.vocab)
    # Pattern for transitive and intransitive verb phrases
    transitive_pattern = [{"POS": "VERB"}, {"POS": "DET", "OP": "?"}, {"POS": "ADJ", "OP": "*"}, {"POS": "NOUN", "OP": "+"}]
    # intransitive_pattern = [{"POS": "VERB"}, {"POS": "ADP"}, {"POS": "DET", "OP": "?"}, {"POS": "ADJ", "OP": "*"}, {"POS": "NOUN", "OP": "+"}]

    # Add patterns to matcher
    matcher.add("verb_phrases", [transitive_pattern])
    # matcher.add("verb_phrases", [transitive_pattern, intransitive_pattern])

    matches = matcher(doc)
    for match_id, start, end in matches:
        span_text = doc[start:end].text.lower()
        found_subphrase = False 
        for i in range(len(verb_phrases)): # find if there is a sub-phrase in the list
            phrase = verb_phrases[i]
            if (phrase in span_text) and len(phrase) < len(span_text):
                verb_phrases[i] = span_text
                found_subphrase = True
            
        if not found_subphrase:
            verb_phrases.append(span_text)

    return verb_phrases


# Remove the duplicates
def add_to_set(case_insensitive_set, items):
    for item in items:
        lower_item = item.lower()
        case_insensitive_set[lower_item] = case_insensitive_set.get(lower_item, 0) + 1
    


def get_top_n_items(dictionary, n=150):
    counter = Counter(dictionary)
    return counter.most_common(n)


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_trf")
    with open("/120040051/test_resource/OMCS/omcs-sentences-more-en.txt", 'r') as file1:
        sentences = [line.strip() for line in file1.readlines()]
    assert(len(sentences) == 956544)
    
    subjects = {}
    verb_phrases = {}

    partial = 100000
    sentence_batch = sentences[:partial]
    total_len = len(sentences)
    i = 1
    for sentence in sentences:
        doc = nlp(sentence)
        # Add items to the respective sets
        add_to_set(subjects, extract_action_subjects(doc))
        add_to_set(verb_phrases, extract_verbs_phrase(doc, nlp))
        i += 1

        if (i % 1000 == 0):
            print(f"Processed {i} / {total_len}")
    print(f"Processed {i-1} / {total_len}")

    
    with open('/120040051/test_resource/OMCS/OMCS-SUBJ-Full.json', 'w') as f:
        json.dump(subjects, f)
    with open('/120040051/test_resource/OMCS/OMCS-VERB-Full.json', 'w') as f:
        json.dump(verb_phrases, f)

        # Filter to keep top 150 for each category
    top_subjects = get_top_n_items(subjects, 500)
    top_verb_phrases = get_top_n_items(verb_phrases, 500)

    with open('/120040051/test_resource/OMCS/OMCS-SUBJ-Full-t500.txt', 'w') as file:
        for subject, freq in top_subjects:
            file.write(f"{subject}: {freq}\n")
    print("Subjects writing complete.")

    with open('/120040051/test_resource/OMCS/OMCS-VERB-Full-t500.txt', 'w') as file:
        for vp, freq in top_verb_phrases:
            file.write(f"{vp}: {freq}\n")
    print("Verb phrases writing complete.")
