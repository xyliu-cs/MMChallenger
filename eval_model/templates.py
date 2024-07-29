INSTRUCTBLIP_MCQ = """Question: [[Text]] Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]]. [[Postfix]] Answer:"""
INSTRUCTBLIP_YN = """[[Prefix]][[Text]][[Postfix]] Answer:"""
INSTRUCTBLIP_SA = """[[Prefix]][[Text]][[Postfix]] Short Answer:"""


# https://huggingface.co/docs/transformers/en/model_doc/llava#usage-tips
LLAVA_VICUNA_MCQ = """USER: <image>
Question: [[Text]]
Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]].
[[Postfix]] Answer with the option's letter from the given choices directly. ASSISTANT:"""
LLAVA_VICUNA_YN = """USER: <image>
[[Prefix]][[Text]][[Postfix]]
Answer Yes or No directly. ASSISTANT:"""
LLAVA_VICUNA_SA = """USER: <image>
[[Prefix]][[Text]][[Postfix]]
Answer no more than 5 words. ASSISTANT:"""


OPENAI_MCQ = """Question: [[Text]] Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]]. [[Postfix]] Answer with the option's letter from the given choices directly."""
OPENAI_YN = """[[Prefix]][[Text]][[Postfix]] Answer Yes or No directly."""
OPENAI_SA = """[[Prefix]][[Text]][[Postfix]] Answer no more than 5 words."""



def route_templates(model_type: str) -> dict:
    if model_type == "llava-vicuna":
        model_templates = {"MCQ": LLAVA_VICUNA_MCQ, "YN": LLAVA_VICUNA_YN, "SA": LLAVA_VICUNA_SA}
    elif model_type == "instructblip":
        model_templates = {"MCQ": INSTRUCTBLIP_MCQ, "YN": INSTRUCTBLIP_YN, "SA": INSTRUCTBLIP_SA}
    return model_templates