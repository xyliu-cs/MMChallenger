This section aims to get **[Who]** (_subject phrases_, e.g. "a dog"), **is [doing what]** (_action phrase_, e.g. "eating breakfast") and **at [Where]** (_place phrases_, e.g. "in the house") from the preprocessed sentence list to form 3 concept sets, which are then combined to form concept tuples and context-target triplets.

Particularly, the `extract_place_phrases.py` script involves `allennlp` and `allennlp-models`, so the running steps should be conducted in a more strict manner as it is not compatible with many other packages, e.g. `ipykernel`. Therefore, you are advised to set up a clean environment to avoid potential conflicts:
```bash
conda create -n allennlp-env python=3.8
conda activate allennlp-env
pip install allennlp allennlp-models
cd build_concept_sets
wget -c https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz
python extract_place_phrases.py input_path output_path
```
Or
```bash
HF_ENDPOINT=https://hf-mirror.com python extract_place_phrases.py input_path output_path
```
if no direct connection to hugging face (as required by allennlp source code)