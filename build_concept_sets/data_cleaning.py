import json


TOP_FREQUENCY_PHRASES = 200

with open('../input_data/OMCS/split_100k/OMCS-SUB-100k.json', 'r') as f:
    subject_all = json.load(f)
with open('../input_data/OMCS/split_100k/OMCS-ACT-100k.json', 'r') as f:
    action_all  = json.load(f)
with open('../input_data/OMCS/split_100k/OMCS-PLA-100k.json', 'r') as f:
    place_all = json.load(f)

targets= [subject_all, action_all, place_all]
names = ["SUB", "ACT", "PLA"]
for idx in range(len(targets)):
    target_data = targets[idx]
    target_data_srt = sorted(target_data.items(), key=lambda item: item[1], reverse=True)
    target_data_keep = target_data_srt[:TOP_FREQUENCY_PHRASES]
    fname = f"OMCS-{names[idx]}-100k-T{TOP_FREQUENCY_PHRASES}.json"
    with open(f"../input_data/Concepte_Sets/before_manual/{fname}", "w") as f:
        json.dump(target_data_keep, f, indent=4)


