# postfix be either '' or 'blahblah. '
# this requires [[Text]] have a tailing white space

INSTRUCTBLIP_MCQ = """Question: [[Text]] Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]]. [[Postfix]]Answer:"""
INSTRUCTBLIP_YN = """[[Prefix]][[Text]][[Postfix]]Answer:"""
INSTRUCTBLIP_SA = """[[Prefix]][[Text]][[Postfix]]Short Answer:"""


# https://huggingface.co/docs/transformers/en/model_doc/llava#usage-tips
LLAVA_VICUNA_MCQ = """USER: <image>
Question: [[Text]]
Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]].
[[Postfix]]Answer with the option's letter from the given choices directly. ASSISTANT:"""
LLAVA_VICUNA_YN = """USER: <image>
[[Prefix]][[Text]][[Postfix]]
Answer Yes or No directly. ASSISTANT:"""
LLAVA_VICUNA_SA = """USER: <image>
[[Prefix]][[Text]][[Postfix]]
Answer no more than 5 words. ASSISTANT:"""

# https://github.com/QwenLM/Qwen-VL/blob/aa00ed04091eea5fcdd32985e7915f1c53e7d599/eval_mm/evaluate_vqa.py#L281
# https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/seed_bench/EVAL_SEED.md#how-to-process-video-by-qwen-vl
QWEN_VL_MCQ = """Question: [[Text]]
Options: A. [[OptionA]]
B. [[OptionB]]
C. [[OptionC]]
D. [[OptionD]]
[[Postfix]]Answer:"""
QWEN_VL_YN = """[[Prefix]][[Text]][[Postfix]]Answer:"""
QWEN_VL_SA = """[[Prefix]][[Text]][[Postfix]]Answer:"""


OPENAI_MCQ = """Question: [[Text]]Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]]. [[Postfix]]Answer with the option's letter from the given choices directly."""
OPENAI_YN = """[[Prefix]][[Text]][[Postfix]]Answer Yes or No directly."""
OPENAI_SA = """[[Prefix]][[Text]][[Postfix]]Answer no more than 5 words."""

OPENAI_CHECK_ANS_PREFIX_ACTION = """You are going to check if a model's response is correct based on the provided information. Specifically, you will be provided a [Question], a [Ground truth] answer to that question, and one or more [Model answer] rounds, where each round contains one or more phrases separated with comma. Your job is to analyze each phrase in each model answer round and determine whether it has a similar meaning to the ground truth answer for the question. If the answer is overly generic but relevant (e.g., "standing"), classify it as partially correct. 

Output "Correct" for correct answers, "Acceptable" for partially correct answers, "Incorrect" for incorrect answers. Output one word per line for each model answer round.

Here are two examples:
[Question] What is a officer doing in a police station?
[Ground truth] paddling boat
[Model answer 1] rowing boat
[Model answer 2] sitting
Expect Output:
[Model answer 1] Correct
[Model answer 2] Acceptable

[Question] What is a baby doing in the kitchen?
[Ground truth] eating some grass
[Model answer 1] eating
[Model answer 2] drinking water
Expect Output:
[Model answer 1] Correct
[Model answer 1] Incorrect

[Question] What is a customer doing in a bank?
[Ground truth] chopping potatoes 
[Model answer 1] cooking
[Model answer 2] cutting vegetable
Expect Output:
[Model answer 1] Correct
[Model answer 2] Correct

Here is the info to check:
"""

def route_templates(model_type: str) -> dict:
    if model_type == "llava-vicuna":
        model_templates = {"MCQ": LLAVA_VICUNA_MCQ, "YN": LLAVA_VICUNA_YN, "SA": LLAVA_VICUNA_SA}
    elif model_type == "instructblip":
        model_templates = {"MCQ": INSTRUCTBLIP_MCQ, "YN": INSTRUCTBLIP_YN, "SA": INSTRUCTBLIP_SA}
    elif model_type == "qwen-vl":
        model_templates = {"MCQ": QWEN_VL_MCQ, "YN": QWEN_VL_YN, "SA": QWEN_VL_SA}
    return model_templates