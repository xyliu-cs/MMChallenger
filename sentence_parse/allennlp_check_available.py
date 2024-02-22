from allennlp.predictors.predictor import Predictor



# Load the pre-trained Semantic Role Labeling model
# predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")
predictor = Predictor.from_path("./bert-base-srl-2020.11.19.tar.gz")

# Example sentence
sentence = "The man is eating pizza in the restaurant on a Sunday afternoon, which is good for him."

# Perform Semantic Role Labeling
result = predictor.predict(sentence)
print(result)


