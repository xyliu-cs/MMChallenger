For the `allennlp` and `allennlp-models`, the processing steps should be conducted with more care as it becomes not compatible with many other packages, e.g. `ipykernel`.

Here are my steps to use `bert-base-srl-2020` model to do SRL:
```bash
conda create -n allennlp-env python=3.8
conda activate allennlp-env
pip install allennlp
pip install allennlp-models
cd test_prior/sentence_parse
wget https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz
python allennlp_srl_parse_loc.py input_path output_path
```