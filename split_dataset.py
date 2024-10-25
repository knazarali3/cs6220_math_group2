import os
os.environ["HF_HOME"] = "~/scratch/hf-cache"

import transformers
import torch
from datasets import load_dataset, DatasetDict, load_from_disk


def create_train_test_val_splits():
    sat_math_dataset = load_dataset("ndavidson/sat-math-chain-of-thought")['train']  # only train available

    sat_math_dataset_correct_answers_only = sat_math_dataset.filter(lambda example: example["is_correct"] == True)
    print(f"There are {len(sat_math_dataset_correct_answers_only)} correct questions and answers")

    train_testvalid = sat_math_dataset_correct_answers_only.train_test_split(test_size=0.1) # 90% train, 10% test + validation

    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)  # Split the 10% test + validation in half test, half valid

    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})

    train_test_valid_dataset.save_to_disk('sat-math-datasets-splits')

def load_train_test_splits():
    sat_math_datasets_splits = load_from_disk('sat-math-datasets-splits')
    return sat_math_datasets_splits

if __name__ == "__main__":
    sat_math_datasets_splits = load_train_test_splits() 
    print(sat_math_datasets_splits)
