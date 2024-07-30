import json, base64

def read_json(json_path):
    print(f"Read json file from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data[:]


def write_json(json_path, json_list):
    with open(json_path, 'w') as f:
        json.dump(json_list, f, indent=2)
    print(f"Write {len(json_list)} items to {json_path}")


def infer_target_type(image_name):
    if image_name.startswith("A"):
        target = "action"
    elif image_name.startswith("P"):
        target = "place"
    else:
        raise ValueError(f"Illegal image naming {image_name}")
    return target

def base64_encode_img(image_path):
    with open(image_path, "rb") as image_file:
        b64_img = base64.b64encode(image_file.read()).decode('utf-8')
    return b64_img