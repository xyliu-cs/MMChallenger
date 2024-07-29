import copy
from lemminflect import getLemma, getInflection


def match_aux_verb(subject: str) -> str:
    words = subject.split()
    key_article = words[0]
    key_noun = words[-1]
    if key_article == 'a': # high priority
        return "is"
    singular_set = set()
    lemma_tup = getLemma(key_noun, upos='NOUN')
    for lemma in lemma_tup:
        lemma_singular_tup = getInflection(lemma, 'NN', False)
        for lemma_singular in lemma_singular_tup:
            singular_set.add(lemma_singular)
    if key_noun in singular_set:
        return "is"
    else:
        return "are"


# gt_triplets: [subject, action, place]
# if a postfix is given, it should include a leading white space
def generate_mcq_prompt(gt_triplet, mcq_dict, target, mcq_prompt_template, postfix=''):
    subject, action, place = gt_triplet[0], gt_triplet[1], gt_triplet[2]
    mcq_tmp = copy.deepcopy(mcq_prompt_template)
    aux = match_aux_verb(subject)
    if target == "place":
        text = f"Where {aux} {subject} {action}?"
    elif target == "action":
        text = f"What {aux} {subject} doing {place}?"
    else:
        raise ValueError(f"Unsupported target type {target}")
    text = text + ' '  
    option_A = mcq_dict["A"]
    option_B = mcq_dict["B"]
    option_C = mcq_dict["C"]
    option_D = mcq_dict["D"]
    mcq_tmp = mcq_tmp.replace("[[Text]]", text, 1) \
                     .replace("[[OptionA]]", option_A, 1) \
                     .replace("[[OptionB]]", option_B, 1) \
                     .replace("[[OptionC]]", option_C, 1) \
                     .replace("[[OptionD]]", option_D, 1) \
                     .replace("[[Postfix]]", postfix, 1)
    return mcq_tmp


# if a postfix is given, it should include a leading white space
# if a prefix is given, it should include a tailing white space
def generate_yn_prompt(gt_triplet, yn_prompt_template, target, postfix='', prefix='', short_ver=False):
    subject, action, place = gt_triplet[0], gt_triplet[1], gt_triplet[2]
    yn_tmp = copy.deepcopy(yn_prompt_template)
    aux = match_aux_verb(subject)
    if short_ver:
        if target == "place":
            text = f"{aux.capitalize()} {subject} {place}?"
        elif target == "action":
            text = f"{aux.capitalize()} {subject} {action}?"
        else:
            raise ValueError(f"Unsupported target type {target}")  
    else:
        if target == "place":
            text = f"{aux.capitalize()} {subject} {action} {place}?"
        elif target == "action":
            text = f"{aux.capitalize()} {subject} {place} {action}?"
        else:
            raise ValueError(f"Unsupported target type {target}")  
    if prefix:
        text = text.lower()
    text = text + ' '
    yn_tmp = yn_tmp.replace("[[Text]]", text, 1) \
                   .replace("[[Prefix]]", prefix, 1) \
                   .replace("[[Postfix]]", postfix, 1)
    
    return yn_tmp


# if a postfix is given, it should include a leading white space
# if a prefix is given, it should include a tailing white space
def generate_sa_prompt(gt_triplet, sa_prompt_template, target, postfix='', prefix=''):
    subject, action, place = gt_triplet[0], gt_triplet[1], gt_triplet[2]
    sa_tmp = copy.deepcopy(sa_prompt_template)
    aux = match_aux_verb(subject)
    if target == "place":
        text = f"Where {aux} {subject} {action}?"
    elif target == "action":
        text = f"What {aux} {subject} doing {place}?"
    else:
        raise ValueError(f"Unsupported target type {target}")  

    if prefix:
        text = text.lower()
    text = text + ' '
    sa_tmp = sa_tmp.replace("[[Text]]", text, 1) \
                   .replace("[[Prefix]]", prefix, 1) \
                   .replace("[[Postfix]]", postfix, 1)
    return sa_tmp