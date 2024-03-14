from lemminflect import getInflection, getLemma
# import argparse

# parser = argparse.ArgumentParser(description='Change the verb in the verb phrases to the present continuous form.')
# parser.add_argument('input_file_path', type=str, help='The path to the input file (verb_phrases_filtered.txt)')
# parser.add_argument('output_file_path', type=str, help='The path to the output file (gerund_phrases_filtered.txt)')

# Execute the parse_args() method
# args = parser.parse_args()
input_file_path = '../raw_results/omcs/filtered_v2/verbs_filtered_v2.txt'
output_file_path = '../raw_results/omcs/filtered_v2/gerunds_filtered_v2.txt'

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

with open(input_file_path, 'r') as f:
    phrases = f.readlines()

# vp_list = [phrase.split(": ")[0] for phrase in phrases]
vp_list = [verb.strip() for verb in phrases]
    
with open(output_file_path, 'w') as f:
    for vp in vp_list:
        words = vp.split()
        verb_ing = get_present_continuous(words[0])
        words[0] = verb_ing
        gerunds = ' '.join(words)
        f.write(gerunds + '\n')