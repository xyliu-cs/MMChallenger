import torch
from transformers import AutoProcessor, CLIPModel
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("/120040051/clip-vit-large-patch14-336").to(device)
processor = AutoProcessor.from_pretrained("/120040051/clip-vit-large-patch14-336")


image_folder = '/120040051/merged0728'
image_file = 'P00424_a_prisoner_taking_a_shower_in_a_board_room.jpg'
image_path = os.path.join(image_folder, image_file)

image = Image.open(image_path).convert("RGB")

text_caps = ["a prisoner taking a shower in a board room", "a prisoner taking a shower in jail"]

inputs = processor(
    text=text_caps, images=image, return_tensors="pt", padding=True
)

inputs = inputs.to(device)
outputs = model(**inputs)
print(outputs.logits_per_image)





class llava_feature_probe:
    def __init__(self, config: dict) -> None:
        self.pretrained_clip_model = CLIPModel.from_pretrained(config["clip_model_path"])
        self.pretrained_clip_processor = AutoProcessor.from_pretrained(config["clip_model_path"])
        self.llava = CLIPModel.from_pretrained(config["clip_model_path"])
        self.clip_model = CLIPModel.from_pretrained(config["clip_model_path"])
