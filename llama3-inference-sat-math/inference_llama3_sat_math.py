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
import re
import ast
import pandas as pd
from tqdm import tqdm
import torch

import os
os.environ["HF_HOME"] = "~/scratch/hf-cache"

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

    """

    def formatting_prompts_func(examples):
        return {"text": alpaca_prompt.format(examples["question"]) }

    return train_dataset.map(formatting_prompts_func)  # Process examples individually

def sat_math_inference():
    
    max_seq_length = 2048 # supports RoPE Scaling internally
    dtype = None # None for auto detection
    load_in_4bit = True # Use 4bit quantization to reduce memory usage.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "knazarali3/llama3_sat_math", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        token = "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"
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    train_dataset, test_dataset = format_reasoning_chain()
    dataset = format_sat_math(test_dataset, tokenizer)

    # Parameters
    batch_size = 10  # Number of questions per batch

    # Initialize a DataFrame to store results
    results_df = pd.DataFrame(columns=["ID", "Question", "Output", "Run_1"])

    # Process the dataset in batches of `batch_size`
    for start_idx in tqdm(range(0, len(dataset["text"]), batch_size), desc="Processing Batches"):
        # Select the current batch of questions and their corresponding IDs
        batch_questions = dataset["text"][start_idx:start_idx + batch_size]
        batch_ids = dataset["id"][start_idx:start_idx + batch_size]  # Assuming 'id' is the column name for question IDs
        
        # Tokenize and generate outputs in batch
        inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                temperature=1e-5,  # For deterministic behavior
                top_k=None,
                top_p=None,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                max_time=300
            )
        
        # Decode all outputs in the batch
        batch_decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print(len(batch_decoded_outputs))

        # Extract and process results for each question
        for question, output, question_id in zip(batch_questions, batch_decoded_outputs, batch_ids):
            response_start = output.find("### Response:") + len("### Response:")
            response = output[response_start:].strip()
            response = response.replace("<|end_of_text|>", "").strip()

            try:
                response_dict = ast.literal_eval(response)
                final_answer = response_dict.get("final_answer", "N/A")
            except (ValueError, SyntaxError):
                final_answer = "Error Parsing Response"

            # Append results to the DataFrame
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame([{
                        "ID": question_id,  # Include the ID column
                        "Question": question,
                        "Output": output,
                        "Run_1": final_answer,
                    }])
                ],
                ignore_index=True
            )


    results_df.to_csv("inference_llama3_sat_math.csv", index=False)

if __name__ == "__main__":
   sat_math_inference()