from sentence_transformers import SentenceTransformer
from question_generator.generate_question import clean_generate
from question_generator.extract_base_sents import extract_base_sentences


if __name__ == '__main__':
# ---------------------------------
# Stage 0 (optional): Preprocessing 
# ---------------------------------
# Input:  Language Corpus
# Output: Cleaned Corpus




# ---------------------------------
# Stage 1 : Parsing Sentences 
# ---------------------------------
# Input:  Clean Language Corpus
# Output: 3 phrase lists of [Subjects] [Verb Phrase] [Location Phrase], 
#         and a complete list of sentences by using Cartisan Product on
#         [Subjects] X [Verb Phrase] X [Location Phrase]




# ---------------------------------
# Stage 2 : Evaluating Perplexity 
# ---------------------------------
# Input:  The complete list of sentences
# Output: The complete list of sentences with perplexity (averaged by the tokens)





# ---------------------------------
# Stage 3 : Question Generation
# ---------------------------------
# Input:  The complete list of sentences with perplexity
# Output: Textual questions with visual images of given type(s)
    print("---------------------------------")
    print("Stage 3 : Question Generation")
    print("---------------------------------")


    sentence_ppl_lookup = "/home/liu/test_resources/sentence_lookup/sentence_lookup_vicuna_2024-03-14_auto.json"
    base_sents_path = "/home/liu/test_resources/base_sentences/base_sentences_0317_reservoir.json"
    ban_words = {'verb': ['riding a bicycle'], 'loc':[]}

    extract_base_sentences(sentence_ppl_lookup, base_sents_path, ban_words, method='reservoir', preview=True)
    model = SentenceTransformer('whaleloops/phrase-bert')

    verb_question = '/home/liu/test_resources/input_questions/verb_questions_0317_t.json'
    loc_question = '/home/liu/test_resources/input_questions/location_questions_0317_t.json'

    clean_generate(sentence_ppl_lookup, base_sents_path, verb_question, 'verb', model, 'reservoir', option_num=5, num_distractors=0, ban_list=ban_words, dump_intermediate=True)
    clean_generate(sentence_ppl_lookup, base_sents_path, loc_question, 'location', model, 'reservoir', option_num=5, num_distractors=0, ban_list=ban_words, dump_intermediate=True)





# ---------------------------------
# Stage 4 : VQA Inference 
# ---------------------------------
# Input:  Textual questions with visual images of given type(s)
# Output: Model's responses






# ---------------------------------
# Stage 5 : Evaluate results
# ---------------------------------
# Input:  Model's responses
# Output: Results analysis