import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import argparse
import json


parser = argparse.ArgumentParser(description='Preprocess the input files.')
parser.add_argument('model_file_path', type=str, help='The path to the hf llama-2 model and tokenizer folder')
parser.add_argument('input_file_path', type=str, help='The path to the compositional sentences json file')
parser.add_argument('--output_file_path', type=str, default='sentence_lookups.json' , help='The path to the output file for sentence ppl lookup')

# Execute the parse_args() method
args = parser.parse_args()
model_file_path = args.model_file_path
input_file_path = args.input_file_path
output_file_path = args.output_file_path


def calculate_perplexity(sentence, tokenizer, model, device):
    model.eval() 
    # Tokenize the input sentence
    encodings = tokenizer(sentence, return_tensors="pt")

    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()
    attention_mask = encodings.attention_mask.to(device)

    # Calculate Negative Log-Likelihood
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids, attention_mask=attention_mask)
        neg_log_likelihood = outputs.loss * input_ids.size(1)  # Multiply by sequence length

    # Calculate Perplexity and append to list
    ppl = torch.exp(neg_log_likelihood / input_ids.size(1))  # Divide by sequence length
    perplexity = ppl.item()

    return perplexity


device = "cuda" if torch.cuda.is_available() else "cpu"
model_folder_name = model_file_path.split('/')[-1].lower()

if "llama2" in model_folder_name:
    tokenizer = LlamaTokenizer.from_pretrained(model_file_path)
    model = LlamaForCausalLM.from_pretrained(model_file_path).to(device)
elif "vicuna" in model_folder_name:
    tokenizer = AutoTokenizer.from_pretrained(model_file_path)
    model = AutoModelForCausalLM.from_pretrained(model_file_path).to(device)
else:
    print(f"Language model {model_folder_name} is not supported yet.")


print(f"Loading sentences from {input_file_path} ...")
with open(input_file_path, 'r') as f:
    j_data = json.load(f)

print(f"Testing the availablity of output path {output_file_path} ...")
with open(output_file_path, 'w') as f:
    j_test = {'a': 2024}
    json.dump(j_test, f)


sent_list = j_data["data"]
sent_list_ppl = []

i = 0
for sent_dict in sent_list:
    sub = sent_dict["subj"]
    text = sent_dict["text"]
    ppl = calculate_perplexity(text, tokenizer, model, device)
    sent_dict["ppl"] = round(ppl, 2)
    sent_list_ppl.append(sent_dict)
    i += 1
    if (i % 100 == 0):
        print(f"Processed {i} sentences")
print(f"\n============\nTotal: Processed {i} sentences\n============")
j_data["data"] = sent_list_ppl


print(f"Writing the results to {output_file_path}, ...")
with open(output_file_path, 'w') as f:
    json.dump(j_data, f)