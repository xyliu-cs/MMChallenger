This section aims to get **[Who]** (_noun phrases_, e.g. "a dog"), **is [doing what]** (_verb phrase_, e.g. "eating breakfast") and **at [Where]** (_positional phrases_, e.g. "in the house") from the preprocessed sentence list to form 3 lists, and then compose the sentences with the format of **[NP] [VP] [PP]**.

Particularly, for the `allennlp_get_positional_phrases.py` script that involves `allennlp` and `allennlp-models`, the running steps should be conducted with more care as it is not compatible with many other packages, e.g. `ipykernel`.

Therefore, you are advised to set up a clean environment to avoid potential conflicts:
```bash
conda create -n allennlp-env python=3.8
conda activate allennlp-env
pip install allennlp
pip install allennlp-models
cd test_prior/sentence_parse
wget https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz
python allennlp_srl_parse_loc.py input_path output_path
```