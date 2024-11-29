import torch
import transformers
from transformers import TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset

from unsloth import FastLanguageModel
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template
from unsloth import is_bfloat16_supported

from tqdm import tqdm
import random
import pandas as pd

import os
os.environ["HF_HOME"] = "~/scratch/hf-cache"


max_seq_length = 2048 # supports RoPE Scaling internally
dtype = None # None for auto detection
load_in_4bit = True # Use 4bit quantization to reduce memory usage.

random.seed(42)

def format_reasoning_chain():

    sat_math_dict = load_dataset("knazarali3/group2_processed_sat_math_cot")
    train_dataset = sat_math_dict["train"]
    test_dataset = sat_math_dict["test"]
    
    dataset_df = pd.DataFrame(train_dataset)

    answers_with_steps = []

    for i in range(len(dataset_df)):
        correct_reasoning_chain = dataset_df["correct_reasoning_chain"][i]
        final_answer = correct_reasoning_chain['final_answer']
        correct_cot_steps = correct_reasoning_chain['steps']
        
        answer_with_steps = {}
        answer_with_steps["steps"] = correct_cot_steps
        answer_with_steps["final_answer"] = final_answer

        keys = list(answer_with_steps.keys())
        keys.sort(reverse=True)

        # Sorted Dictionary
        answer_with_steps = {i: answer_with_steps[i] for i in keys}

        answers_with_steps.append(str(answer_with_steps))

    dataset_df.loc[: , "answers_with_steps"] = answers_with_steps
    train_dataset = Dataset.from_pandas(dataset_df)

    dataset_df = pd.DataFrame(test_dataset)

    answers_with_steps = []

    for i in range(len(dataset_df)):
        correct_reasoning_chain = dataset_df["correct_reasoning_chain"][i]
        final_answer = correct_reasoning_chain['final_answer']
        correct_cot_steps = correct_reasoning_chain['steps']
        
        answer_with_steps = {}
        answer_with_steps["steps"] = correct_cot_steps
        answer_with_steps["final_answer"] = final_answer

        keys = list(answer_with_steps.keys())
        keys.sort(reverse=True)

        # Sorted Dictionary
        answer_with_steps = {i: answer_with_steps[i] for i in keys}

        answers_with_steps.append(str(answer_with_steps))
        
    dataset_df.loc[: , "answers_with_steps"] = answers_with_steps
    test_dataset = Dataset.from_pandas(dataset_df)

    return train_dataset, test_dataset




def format_sat_math(train_dataset, tokenizer):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    The input contains a math problem. Carefully solve the problem step-by-step with only logical/mathematical steps. 

    Please provide the answer in the following format: A steps key containing a list of step-by-step calculations and explanations. A final_answer key with the direct answer
    
    ### Input:
    {}

    ### Response:
    {}"""

    def formatting_prompts_func(examples):
        return {"text": alpaca_prompt.format(examples["question"], examples["answers_with_steps"]) + tokenizer.eos_token }

    return train_dataset.map(formatting_prompts_func)  # Process examples individually



def train_llama3_sat_math():
    sat_math_dict = load_dataset("knazarali3/group2_processed_sat_math_cot")
    train_dataset, test_dataset = format_reasoning_chain()


    # initalize model and tokenizer 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B", # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    formatted_sat_math_train_data = format_sat_math(train_dataset, tokenizer)

    # train 
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_sat_math_train_data,
        #dataset_text_field = "text",
        #max_seq_length = max_seq_length,
        #dataset_num_proc = 2,
        #packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 10,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 2, 
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )

    trainer_stats = trainer.train()

    model.push_to_hub("knazarali3/llama3_SAT_MATH", token = "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") # Online saving
    tokenizer.push_to_hub("knazarali3/llama3_SAT_MATH", token = "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") # Online saving


if __name__ == "__main__":
    train_llama3_sat_math()