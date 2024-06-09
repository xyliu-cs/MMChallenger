import spacy
from collections import Counter
import json

def extract_subject_phrases(doc):
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


def extract_action_phrases(doc):
    action_phrases = []

    for token in doc:
        if token.pos_ == "VERB":
            # Find the direct object child of the verb
            direct_objs = [child for child in token.children if child.dep_ == "dobj"]
            for dobj in direct_objs:
                # Include only determiners and compound nouns in the phrase
                det = [det.text for det in dobj.lefts if det.dep_ == 'det']
                compounds = [comp.text for comp in dobj.lefts if comp.dep_ == 'compound']  # omit multi-layered compound

                # Construct the direct object phrase excluding any adjectives
                dobj_phrase = ' '.join(det + compounds + [dobj.text])

                # Build the verb phrase by combining the verb with the direct object phrase
                verb_phrase = f"{token.text} {dobj_phrase}"
                verb_phrases.append(verb_phrase.lower())

    return action_phrases


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
        add_to_set(subjects, extract_subject_phrases(doc))
        add_to_set(verb_phrases, extract_action_phrases(doc))
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
