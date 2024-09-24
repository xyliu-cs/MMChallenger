# postfix be either '' or 'blahblah. '
# this requires [[Text]] have a tailing white space

INSTRUCTBLIP_MCQ = """Question: [[Text]] Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]]. [[Postfix]]Answer:"""
INSTRUCTBLIP_YN = """[[Prefix]][[Text]][[Postfix]]Answer:"""
INSTRUCTBLIP_SA = """[[Prefix]][[Text]][[Postfix]]Short Answer:"""


INSTRUCTBLIP_MCQ_COT = """Question: [[Text]] Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]]. [[Postfix]]Let's think step by step and provide the final answer as an option letter at last after saying "The final answer:". First,"""
INSTRUCTBLIP_YN_COT = """[[Prefix]][[Text]][[Postfix]]Let's think step by step and provide the final answer at last after saying "The final answer:"."""
INSTRUCTBLIP_SA_COT = """[[Prefix]][[Text]][[Postfix]]Let's think step by step and provide a short answer at last after saying "The final answer:". First,"""


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


LLAVA_VICUNA_MCQ_COT = """USER: <image>
Question: [[Text]]
Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]].
[[Postfix]]Provide your final answer with the option's letter from the given choices directly at last after saying 'Therefore, the answer is'. ASSISTANT: Let's think step by step."""
LLAVA_VICUNA_YN_COT = """USER: <image>
[[Prefix]][[Text]][[Postfix]]Provide your final answer with Yes or No directly at last after saying 'Therefore, the answer is'.
ASSISTANT: Let's think step by step."""
LLAVA_VICUNA_SA_COT = """USER: <image>
[[Prefix]][[Text]][[Postfix]]Provide your final answer within 5 words at last after saying 'Therefore, the answer is'.
ASSISTANT: Let's think step by step."""


LLAVA_VICUNA_MCQ_ST = """USER: Question: [[Text]]
Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]].
[[Postfix]]Answer with the option's letter from the given choices directly. \nASSISTANT:"""
LLAVA_VICUNA_YN_ST = """USER: [[Prefix]][[Text]][[Postfix]]Answer Yes or No directly. \nASSISTANT:"""
LLAVA_VICUNA_SA_ST = """USER: [[Prefix]][[Text]][[Postfix]]Answer no more than 5 words. \nASSISTANT:"""


# https://huggingface.co/docs/transformers/main/en/model_doc/llava_next#usage-tips
LLAVA_LLAMA3_MCQ = """<|start_header_id|>system<|end_header_id|>

You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>

<image>
Question: [[Text]]
Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]].
[[Postfix]]Answer with the option's letter from the given choices directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
LLAVA_LLAMA3_YN = """<|start_header_id|>system<|end_header_id|>

You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>

<image>
[[Prefix]][[Text]][[Postfix]]Answer Yes or No directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
LLAVA_LLAMA3_SA = """<|start_header_id|>system<|end_header_id|>

You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>

<image>
[[Prefix]][[Text]][[Postfix]]Answer no more than 5 words.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""



# https://huggingface.co/docs/transformers/main/en/model_doc/llava_next#usage-tips
LLAVA_LLAMA3_MCQ_COT = """<|start_header_id|>system<|end_header_id|>

You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>

<image>
Question: [[Text]]
Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]].
[[Postfix]]Provide your final answer with the option's letter from the given choices at last after saying "Therefore, the answer is".<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step."""
LLAVA_LLAMA3_YN_COT = """<|start_header_id|>system<|end_header_id|>

You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>

<image>
[[Prefix]][[Text]][[Postfix]]Provide your final answer in one word at last after saying "Therefore, the answer is".<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step."""
LLAVA_LLAMA3_SA_COT = """<|start_header_id|>system<|end_header_id|>

You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>

<image>
[[Prefix]][[Text]][[Postfix]]Provide your final answer in 5 words at last after saying "Therefore, the answer is".<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet's think step by step."""


# https://huggingface.co/docs/transformers/main/en/model_doc/llava_next#usage-tips
LLAVA_LLAMA3_MCQ_ST = """<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>

Question: [[Text]]
Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]].
[[Postfix]]Answer with the option's letter from the given choices directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
LLAVA_LLAMA3_YN_ST = """<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>

[[Prefix]][[Text]][[Postfix]]Answer Yes or No directly.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
LLAVA_LLAMA3_SA_ST = """<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>

[[Prefix]][[Text]][[Postfix]]Answer no more than 5 words.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""


# https://huggingface.co/docs/transformers/main/en/model_doc/llava_next#usage-tips
LLAVA_YI_MCQ = """<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>
Question: [[Text]]
Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]].
[[Postfix]]Answer with the option's letter from the given choices directly.<|im_end|><|im_start|>assistant\n"""
LLAVA_YI_YN = """<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>
[[Prefix]][[Text]][[Postfix]]Answer Yes or No directly.<|im_end|><|im_start|>assistant\n"""
LLAVA_YI_SA = """<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>
[[Prefix]][[Text]][[Postfix]]Answer no more than 5 words.<|im_end|><|im_start|>assistant\n"""


LLAVA_YI_MCQ_COT = """<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>
Question: [[Text]]
Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]].
[[Postfix]]Provide your final answer with the option's letter from the given choices at last after saying 'Therefore, the answer is'.<|im_end|><|im_start|>assistant\nLet's think step by step."""
LLAVA_YI_YN_COT = """<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>
[[Prefix]][[Text]][[Postfix]]Provide your final answer 'Yes' or 'No' at last after saying 'Therefore, the answer is'.<|im_end|><|im_start|>assistant\nLet's think step by step."""
LLAVA_YI_SA_COT = """<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>
[[Prefix]][[Text]][[Postfix]]Provide your final answer within 5 words at last after saying 'Therefore, the answer is'.<|im_end|><|im_start|>assistant\nLet's think step by step."""


# https://huggingface.co/docs/transformers/main/en/model_doc/llava_next#usage-tips
LLAVA_YI_MCQ_ST = """<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user
Question: [[Text]]
Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]].
[[Postfix]]Answer with the option's letter from the given choices directly.<|im_end|><|im_start|>assistant\n"""
LLAVA_YI_YN_ST = """<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user
[[Prefix]][[Text]][[Postfix]]Answer Yes or No directly.<|im_end|><|im_start|>assistant\n"""
LLAVA_YI_SA_ST = """<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user
[[Prefix]][[Text]][[Postfix]]Answer no more than 5 words.<|im_end|><|im_start|>assistant\n"""

# https://github.com/QwenLM/Qwen-VL/blob/aa00ed04091eea5fcdd32985e7915f1c53e7d599/eval_mm/evaluate_vqa.py#L281
# https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/seed_bench/EVAL_SEED.md#how-to-process-video-by-qwen-vl
# https://github.com/QwenLM/Qwen-VL/blob/aa00ed04091eea5fcdd32985e7915f1c53e7d599/eval_mm/evaluate_multiple_choice.py#L128
QWEN_VL_MCQ = """Question: [[Text]]
Options: A. [[OptionA]]
B. [[OptionB]]
C. [[OptionC]]
D. [[OptionD]]
[[Postfix]]Answer:"""
QWEN_VL_YN = """[[Prefix]][[Text]][[Postfix]]Answer:"""
QWEN_VL_SA = """[[Prefix]][[Text]][[Postfix]]Answer:"""


QWEN_VL_MCQ_COT = """Question: [[Text]]
Options: A. [[OptionA]]
B. [[OptionB]]
C. [[OptionC]]
D. [[OptionD]]
[[Postfix]]Provide your final answer at last after saying "Therefore, the answer is" """
QWEN_VL_YN_COT = """[[Prefix]][[Text]][[Postfix]]Provide your final answer as "Yes" or "No" at last after saying "Therefore, the answer is" """
QWEN_VL_SA_COT = """[[Prefix]][[Text]][[Postfix]]Provide your final answer briefly after saying "Therefore, the answer is" """

QWEN_VL_MCQ_ST = """Question: [[Text]]
Options: A. [[OptionA]]
B. [[OptionB]]
C. [[OptionC]]
D. [[OptionD]]
[[Postfix]]Choice Answer:"""
QWEN_VL_YN_ST = """[[Prefix]][[Text]][[Postfix]]Answer:"""
QWEN_VL_SA_ST = """[[Prefix]][[Text]][[Postfix]]Answer:"""


QWEN_VL_CHAT_MCQ = """Question: [[Text]]
Options: A. [[OptionA]]
B. [[OptionB]]
C. [[OptionC]]
D. [[OptionD]]
[[Postfix]]Answer with the option's letter from the given choices directly."""
QWEN_VL_CHAT_YN = """[[Prefix]][[Text]][[Postfix]]Answer Yes or No directly."""
QWEN_VL_CHAT_SA = """[[Prefix]][[Text]][[Postfix]]Answer with a single word or phrase."""


OPENAI_MCQ = """Question: [[Text]]Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]]. [[Postfix]]Answer with the option's letter from the given choices directly."""
OPENAI_YN = """[[Prefix]][[Text]][[Postfix]]Answer Yes or No directly."""
OPENAI_SA = """[[Prefix]][[Text]][[Postfix]]Answer within 5 words."""


OPENAI_MCQ_COT = """Question: [[Text]]Options: (A) [[OptionA]]. (B) [[OptionB]]. (C) [[OptionC]]. (D) [[OptionD]]. [[Postfix]]Think step-by-step and give your the final answer as an option letter at last after saying "Final Answer:"."""
OPENAI_YN_COT = """[[Prefix]][[Text]][[Postfix]]Think step-by-step and give your the final answer as Yes or No at last after saying "Final Answer:"."""
OPENAI_SA_COT = """[[Prefix]][[Text]][[Postfix]]Think step-by-step and give your the final answer within 5 words at last after saying "Final Answer:"."""


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


def route_templates(model_type: str, sanity_test: bool, use_cot: bool) -> dict:
    assert not (sanity_test and use_cot), "sanity_test and use_cot cannot be true at the same time"
    if model_type in ["llava-vicuna"]:
        if sanity_test:
            model_templates = {"MCQ": LLAVA_VICUNA_MCQ_ST, "YN": LLAVA_VICUNA_YN_ST, "SA": LLAVA_VICUNA_SA_ST}
        elif use_cot:
            model_templates = {"MCQ": LLAVA_VICUNA_MCQ_COT, "YN": LLAVA_VICUNA_YN_COT, "SA": LLAVA_VICUNA_SA_COT}
        else:
            model_templates = {"MCQ": LLAVA_VICUNA_MCQ, "YN": LLAVA_VICUNA_YN, "SA": LLAVA_VICUNA_SA}
    elif model_type == "llava-llama3":
        if sanity_test:
            model_templates = {"MCQ": LLAVA_LLAMA3_MCQ_ST, "YN": LLAVA_LLAMA3_YN_ST, "SA": LLAVA_LLAMA3_SA_ST}
        elif use_cot:
            model_templates = {"MCQ": LLAVA_LLAMA3_MCQ_COT, "YN": LLAVA_LLAMA3_YN_COT, "SA": LLAVA_LLAMA3_SA_COT}
        else:
            model_templates = {"MCQ": LLAVA_LLAMA3_MCQ, "YN": LLAVA_LLAMA3_YN, "SA": LLAVA_LLAMA3_SA}
    elif model_type == "llava-yi":
        if sanity_test:
            model_templates = {"MCQ": LLAVA_YI_MCQ_ST, "YN": LLAVA_YI_YN_ST, "SA": LLAVA_YI_SA_ST}
        elif use_cot:
            model_templates = {"MCQ": LLAVA_YI_MCQ_COT, "YN": LLAVA_YI_YN_COT, "SA": LLAVA_YI_SA_COT}
        else:
            model_templates = {"MCQ": LLAVA_YI_MCQ, "YN": LLAVA_YI_YN, "SA": LLAVA_YI_SA}
    elif model_type in ["instructblip-vincuna", "blip2-t5", "instructblip-t5"]:
        if use_cot:
            model_templates = {"MCQ": INSTRUCTBLIP_MCQ_COT, "YN": INSTRUCTBLIP_YN_COT, "SA": INSTRUCTBLIP_SA_COT}
        else: # sanity test is same as non-cot
            model_templates = {"MCQ": INSTRUCTBLIP_MCQ, "YN": INSTRUCTBLIP_YN, "SA": INSTRUCTBLIP_SA}
    elif model_type == "qwen-vl-chat":
        if use_cot:
            model_templates = {"MCQ": QWEN_VL_MCQ_COT, "YN": QWEN_VL_YN_COT, "SA": QWEN_VL_SA_COT}
        else: # sanity test is same as non-cot
            model_templates = {"MCQ": QWEN_VL_CHAT_MCQ, "YN": QWEN_VL_CHAT_YN, "SA": QWEN_VL_CHAT_SA}
    elif model_type == "qwen-vl":
        if sanity_test:
            model_templates = {"MCQ": QWEN_VL_MCQ_ST, "YN": QWEN_VL_YN_ST, "SA": QWEN_VL_SA_ST}
        else:
            model_templates = {"MCQ": QWEN_VL_MCQ, "YN": QWEN_VL_YN, "SA": QWEN_VL_SA}
    elif model_type in ["gpt", "anthropic"]:
        if use_cot:
            model_templates = {"MCQ": OPENAI_MCQ_COT, "YN": OPENAI_YN_COT, "SA": OPENAI_SA_COT}
        else:
            model_templates = {"MCQ": OPENAI_MCQ, "YN": OPENAI_YN, "SA": OPENAI_SA}
    else:
        raise ValueError(f"Unsupported model type {model_type}")
    return model_templates