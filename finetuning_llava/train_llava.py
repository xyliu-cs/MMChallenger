from trl.commands.cli_utils import SFTScriptArguments, TrlParser
from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, LlavaForConditionalGeneration, AutoModelForVision2Seq, BitsAndBytesConfig
import torch
import numpy as np
import warnings
from datasets import load_dataset
from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from accelerate import PartialState
from peft import LoraConfig, get_peft_model

class LLavaDataCollator:
    def __init__(self, processor, response_template):
        self.processor = processor
        self.response_template = response_template
        self.response_token_ids = self.processor.tokenizer.encode(response_template, add_special_tokens=False)

    def __call__(self, examples):
        texts = [self.processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False) for example in examples]
        images = [example["images"] for example in examples]
        mm_batch = self.processor(texts, images, return_tensors="pt", padding=True)
    
        labels = mm_batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        mm_batch["labels"] = labels
        
        # print('len of examples: ', len(examples))
        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(mm_batch["labels"][i] == self.response_token_ids[0])[0]: # tuple of array, get into the first array
                # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if (
                    self.response_token_ids
                    == mm_batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                ):
                    response_token_ids_start_idx = idx
            if response_token_ids_start_idx is None:
                warnings.warn(
                    f"Could not find response key `{self.response_template}` in the "
                    f'following instance: {self.processor.tokenizer.decode(mm_batch["input_ids"][i])} '
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                mm_batch["labels"][i, :] = -100
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)
                # Make pytorch loss function ignore all tokens up through the end of the response key
                mm_batch["labels"][i, :response_token_ids_end_idx] = -100
        return mm_batch


        
if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    sft_script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.dataset_text_field = "text"  # need a dummy field
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    
    # training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    torch.cuda.empty_cache()
    
    raw_datasets = load_dataset(sft_script_args.dataset_name)
    train_dataset = raw_datasets["train"]
    
    print('number of training examples: ', len(train_dataset))
    print('per_device_train_batch_size: ', training_args.per_device_train_batch_size)    
    print('gradient_accumulation_steps: ', training_args.gradient_accumulation_steps)
    print('training epoch: ', training_args.num_train_epochs)
    
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path 
    )
    # device_string = PartialState().process_index
    collator = LLavaDataCollator(processor, response_template="ASSISTANT:")
    
    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    # )
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16, device_map='auto'
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,
        data_collator=collator
    )

    trainer.train()
    # trainer.save_model(training_args.output_dir)
    # trainer.push_to_hub()