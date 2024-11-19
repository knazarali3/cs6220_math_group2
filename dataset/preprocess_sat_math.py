import torch
import transformers
from datasets import load_dataset, DatasetDict, Dataset

import random
import pandas as pd

import os
os.environ["HF_HOME"] = "~/scratch/hf-cache"

def preprocess_sat_math():

    random.seed(42)

    # Only keep correct answers
    dataset = load_dataset("ndavidson/sat-math-chain-of-thought")["train"]
    dataset_df = pd.DataFrame(dataset)
    filtered_dataset_df = dataset_df[dataset_df['is_correct']==True]
    correct_reasoning_chain_list = []

    # Remove wrong steps in the reasoning chain
    for i in filtered_dataset_df.index:
        final_answer = filtered_dataset_df["reasoning_chain"][i]['final_answer']
        correct_cot_steps = [ {"explanation" : item['explanation']} for item in filtered_dataset_df["reasoning_chain"][i]['steps'] if item['is_correct'] == True]

        correct_reasoning_chain = {}
        correct_reasoning_chain["final_answer"] = final_answer
        correct_reasoning_chain["steps"] = correct_cot_steps

        correct_reasoning_chain_list.append(correct_reasoning_chain)
    
    filtered_dataset_df.loc[: , "correct_reasoning_chain"] = correct_reasoning_chain_list

    # Remove duplicate rows and reasoning_chain that has incorrect steps
    filtered_dataset_df = filtered_dataset_df.drop(columns=['reasoning_chain'])
    filtered_dataset_df = filtered_dataset_df.drop_duplicates(subset=['id'])
    dataset = Dataset.from_pandas(filtered_dataset_df)
    
    dataset_train_test = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset_train_test['train']
    test_dataset = dataset_train_test['test']

    train_dataset = train_dataset.rename_column("__index_level_0__", "index")
    test_dataset = test_dataset.rename_column("__index_level_0__", "index")

    #train_dataset.push_to_hub("knazarali3/group2_processed_sat_math_cot", split="train")
    #test_dataset.push_to_hub("knazarali3/group2_processed_sat_math_cot", split="test")


if __name__ == "__main__":
    #preprocess_sat_math()
    
    print("###############  SAT Math Dataset ##################")
    sat_math_dict = load_dataset("knazarali3/group2_processed_sat_math_cot")
    print(sat_math_dict)

    print("###############  Competition Math Dataset ##################")
    competition_math_dict = load_dataset("jeggers/competition_math", "original")
    print(competition_math_dict)

