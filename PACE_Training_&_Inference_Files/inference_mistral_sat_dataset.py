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
os.environ["HF_HOME"] = "~/scratch/CS_6220_Project/hf-cache"

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

def mistral_inference_sat_with_resume():
    max_seq_length = 2048  # supports RoPE Scaling internally
    dtype = None  # None for auto detection
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="SaalihaA/mistral_v7_SAT_Math_Dataset",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token="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"
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    train_dataset, test_dataset = format_reasoning_chain()
    test_dataset = format_sat_math(test_dataset, tokenizer)
    row_index_test_dataset = pd.DataFrame(test_dataset).index
    test_dataset = test_dataset.add_column("row_index", row_index_test_dataset)

    dataset = test_dataset
    # Parameters
    batch_size = 10  # Number of questions per batch
    output_file = "Mistral_Inference_SAT_Dataset.csv"

    # Initialize a DataFrame to store results
    results_df = pd.DataFrame(columns=["row_index", "problem", "Output", "Run_1"])

    # Determine the starting index
    try:
        processed_indices = pd.read_csv(output_file)["row_index"].tolist()
        start_index = max(processed_indices) + 1  # Resume from the next unprocessed index
    except (FileNotFoundError, ValueError):
        start_index = 0  # Start from the beginning if the file doesn't exist or is empty

    # Process the dataset in batches of `batch_size`
    for start_idx in tqdm(range(start_index, len(dataset["text"]), batch_size), desc="Processing Batches"):
        # Select the current batch of questions and their corresponding IDs
        batch_questions = dataset["text"][start_idx:start_idx + batch_size]
        batch_ids = dataset["row_index"][start_idx:start_idx + batch_size] 
        
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
                max_time=100
            )

        # Decode all outputs in the batch
        batch_decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract and process results for each question
        for question, output, question_id in zip(batch_questions, batch_decoded_outputs, batch_ids):
            response_start = output.find("### Response:") + len("### Response:")
            response = output[response_start:].strip()
            response = response.replace("<|end_of_text|>", "").strip()

            try:
                response_dict = ast.literal_eval(response)

                if isinstance(response_dict, dict):
                    final_answer = response_dict.get("final_answer", "N/A")
                else:
                    final_answer = "Error: Response is not a dictionary"
            except (ValueError, SyntaxError):
                final_answer = "Error Parsing Response"

            # Append results to the DataFrame
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame([{
                        "row_index": question_id,  # Include the ID column
                        "problem": question,
                        "Output": output,
                        "Run_1": final_answer,
                    }])
                ],
                ignore_index=True
            )

        # Save results incrementally after each batch
        write_mode = "w" if start_idx == 0 else "a"  # Overwrite only for the first batch
        include_header = start_idx == 0  # Include header only for the first batch
        results_df.to_csv(output_file, mode=write_mode, header=include_header, index=False)

        # Clear the DataFrame after writing to save memory
        results_df = pd.DataFrame(columns=["row_index", "problem", "Output", "Run_1"])

output_file = "Mistral_Inference_SAT_Dataset.csv"
 # Determine the starting index
try:
    processed_indices = pd.read_csv(output_file)["row_index"].tolist()
    start_index = max(processed_indices) + 1  # Resume from the next unprocessed index
except (FileNotFoundError, ValueError):
    start_index = 0  # Start from the beginning if the file doesn't exist or is empty
print(start_index)

if __name__ == "__main__":
   mistral_inference_sat_with_resume()