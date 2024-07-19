import torch, json, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def calculate_log_prob(_model, _tokenizer, data_frame, batch_size):
    print(f"Set batch size to {batch_size}, using device {torch.cuda.get_device_name(torch.cuda.current_device())}")
    if _tokenizer.add_bos_token:
        for data_dict in tqdm(data_frame, "Calculating log probabilities"):
            expr = data_dict["expr"]
            encodings = _tokenizer(expr, return_tensors='pt')
            input_ids = encodings.input_ids
            with torch.no_grad():
                outputs = _model(input_ids, labels=input_ids)
                # labels shift left one place, so the number of labels being averaged on = input labels - 1
                cum_log_likelihood = outputs.loss * -(input_ids.shape[1]-1)
            data_dict["logprob"] = round(cum_log_likelihood.item(), 2)
        return data_frame

# def calculate_masked_loss(_model, _tokenizer, sentence, masked_prefix, device):
#     if _tokenizer.add_bos_token:
#         encodings = _tokenizer(sentence, return_tensors='pt')
#         masked_encodings = _tokenizer(masked_prefix, return_tensors='pt')
#         input_ids = encodings.input_ids.to(device)
#         label_ids = input_ids.clone()
#         masked_len = masked_encodings.input_ids.shape[1]
#         label_ids[: , :masked_len] = -100  # omit their loss
#         label_ids.to(device)
#         with torch.no_grad():
#             # cross-entropy loss being averaged on the valid labels
#             outputs = _model(input_ids, labels=label_ids)
#         return outputs.loss.item()

# def calculate_masked_loss_avg_words(_model, _tokenizer, sentence, masked_prefix, device):
#     if _tokenizer.add_bos_token:
#         encodings = _tokenizer(sentence, return_tensors='pt')
#         masked_encodings = _tokenizer(masked_prefix, return_tensors='pt')
#         input_ids = encodings.input_ids.to(device)
#         label_ids = input_ids.clone()
#         masked_len = masked_encodings.input_ids.shape[1]
#         word_counts = len(sentence[len(masked_prefix)+1:].split())
#         label_ids[: , :masked_len] = -100  # omit their loss
#         label_ids.to(device)
#         with torch.no_grad():
#             # cross-entropy loss being averaged on the valid labels
#             outputs = _model(input_ids, labels=label_ids)
#             cum_log_likelihood = outputs.loss * -(input_ids.shape[1]-masked_len)
#             log_avg_on_words = cum_log_likelihood / word_counts
#         return log_avg_on_words.item()


if __name__ == '__main__':
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained('/120040051/vicuna-13b-v1.5')
    # model = AutoModelForCausalLM.from_pretrained('/120040051/vicuna-7b-v1.5', device_map ='auto')
    model = AutoModelForCausalLM.from_pretrained('/120040051/vicuna-13b-v1.5', device_map ='auto')
    input_file = "context_exprs.json"
    output_file = "context_exprs_scored.json"
    with open(input_file, 'r') as f:
        data = json.load(f)
    data_scored = calculate_log_prob(data[:50], tokenizer, model)
    with open(output_file, 'w') as f:
        json.dumps(data_scored)
