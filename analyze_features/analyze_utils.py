from lemminflect import getInflection, getLemma
from PIL import Image
import json, torch

def read_json(json_path):
    print(f"Read json file from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(json_path: str, json_list: list):
    with open(json_path, 'w') as f:
        json.dump(json_list, f, indent=2)
    print(f"Write {len(json_list)} items to {json_path}")

def read_image(img_path: str) -> Image:
    image_obj = Image.open(img_path).convert("RGB")
    return image_obj

def convert_to_vbg(verb_phrase: str) -> str:
    words = verb_phrase.split()
    if not words[0].endswith('ing'):
        lemma = getLemma(words[0], upos='VERB')
        if lemma:
            v_ing = getInflection(lemma[0], tag='VBG')
            if v_ing:
                words[0] = v_ing[0]
                return ' '.join(words)
    # print(f"Not found VBG from verb {words[0]}")
    return verb_phrase

def cosine_sim(pooled_image_embeds: torch.tensor, pooled_text_embeds: torch.tensor) -> torch.tensor:
    norm_img = pooled_image_embeds / pooled_image_embeds.norm(p=2, dim=-1, keepdim=True)
    norm_text = pooled_text_embeds / pooled_text_embeds.norm(p=2, dim=-1, keepdim=True)
    cosine_sim = torch.matmul(norm_text, norm_img.t().to(pooled_text_embeds.device))
    return cosine_sim.squeeze()

