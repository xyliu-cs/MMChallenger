import evaluate
import json

input_file_path = '/120040051/test_resource/raw_sentences/tmp_compositional_sents_2024-03-14.json'
output_file_path = '/120040051/test_resource/sentence_ppl_lookup/sentence_lookup_vicuna_2024-03-14_auto.json'

perplexity = evaluate.load("perplexity", module_type="metric")
print(f"Loading sentences from {input_file_path} ...")
with open(input_file_path, 'r') as f:
    sent_list = json.load(f)

# sent_list = j_data["data"][:]
sent_list_text = [sent_dict['text'] for sent_dict in sent_list]

ppl_results = perplexity.compute(model_id='/120040051/vicuna-7b-v1.5', add_start_token=False, predictions=sent_list_text)

sent_list_ppl = []
for i in range(len(sent_list_text)):
    # sub = sent_dict["subj"]
    # text = sent_dict["text"]
    # ppl = calculate_perplexity(text, tokenizer, model, device)
    assert (sent_list[i]["text"] == sent_list_text[i])
    sent_list[i]["ppl"] = round(ppl_results['perplexities'][i], 2)
    sent_list_ppl.append(sent_list[i])
    i += 1
    # if (i % 100 == 0):
    #     print(f"Processed {i} sentences")
print(f"\n============\nTotal: Processed {i} sentences\n============")
# j_data["data"] = sent_list_ppl

print(f"Writing the results to {output_file_path}, ...")
with open(output_file_path, 'w') as f:
    json.dump(sent_list_ppl, f)