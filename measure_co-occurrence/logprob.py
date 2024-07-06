import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


def calculate_log_prob(_model, _tokenizer, sentence):
    if _tokenizer.add_bos_token:
        encodings = _tokenizer(sentence, return_tensors='pt')
        input_ids = encodings.input_ids
        with torch.no_grad():
            outputs = _model(input_ids, labels=input_ids)
            # labels shift left one place, so the number of labels being averaged on = input labels - 1
            cum_log_likelihood = outputs.loss * -(input_ids.shape[1]-1)
        return cum_log_likelihood.item()

def calculate_masked_loss(_model, _tokenizer, sentence, masked_prefix, device):
    if _tokenizer.add_bos_token:
        encodings = _tokenizer(sentence, return_tensors='pt')
        masked_encodings = _tokenizer(masked_prefix, return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        label_ids = input_ids.clone()
        masked_len = masked_encodings.input_ids.shape[1]
        label_ids[: , :masked_len] = -100  # omit their loss
        label_ids.to(device)
        with torch.no_grad():
            # cross-entropy loss being averaged on the valid labels
            outputs = _model(input_ids, labels=label_ids)
        return outputs.loss.item()

def calculate_masked_loss_avg_words(_model, _tokenizer, sentence, masked_prefix, device):
    if _tokenizer.add_bos_token:
        encodings = _tokenizer(sentence, return_tensors='pt')
        masked_encodings = _tokenizer(masked_prefix, return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        label_ids = input_ids.clone()
        masked_len = masked_encodings.input_ids.shape[1]
        word_counts = len(sentence[len(masked_prefix)+1:].split())
        label_ids[: , :masked_len] = -100  # omit their loss
        label_ids.to(device)
        with torch.no_grad():
            # cross-entropy loss being averaged on the valid labels
            outputs = _model(input_ids, labels=label_ids)
            cum_log_likelihood = outputs.loss * -(input_ids.shape[1]-masked_len)
            log_avg_on_words = cum_log_likelihood / word_counts
        return log_avg_on_words.item()


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained('/120040051/vicuna-7b-v1.5')
    # model = AutoModelForCausalLM.from_pretrained('/120040051/vicuna-7b-v1.5', device_map ='auto')
    model = AutoModelForCausalLM.from_pretrained('/120040051/vicuna-7b-v1.5').to(device)
    input_file = "/120040051/test_resource/raw_sentences/tmp_compositional_sents_0328.json"
    output_file = "/120040051/test_resource/sentence_ppl_lookup/sents_logprob_avg_0331.json"
    with open(input_file, 'r') as f:
        data = json.load(f)
    new_data = []
    i = 0
    for sent_dict in data:
        new_dict = sent_dict.copy()
        sentence_text = sent_dict['text']
        masked_prefix = sent_dict['subj'] + ' ' + sent_dict['be']
        masked_loss = calculate_masked_loss_avg_words(model, tokenizer, sentence_text, masked_prefix, device)
        # new_dict['logprob'] = round(logprob, 2)
        new_dict['loss'] = round(masked_loss, 2)
        new_data.append(new_dict)
        i += 1
        if (i % 1000 == 0):
            print(f"Processed {i} sentences")
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)