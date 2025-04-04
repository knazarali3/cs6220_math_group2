# CS6220 Big Data Systems and Analytics: Group 2

Instructor: Professor Ling Liu <br>
Team: Kiran Nazarali, Michael Hu, Meredith Rush, Saaliha Allauddin

| Branch Name | Owner | Model link (if applicable)
|------------------|------------------|------------------|
| Analysis_Kiran  | Kiran  | |
| Analysis_Michael  | Michael | |
| GPT   | Michael    | |
| Mistral_7B   | Saaliha    | https://huggingface.co/SaalihaA/mistral_v7_Competition_Dataset |
| QWEN   | Michael    | https://huggingface.co/MichaelHu03/CS6220-QWEN-Comp-Finetuned|
| llama3   | Kiran    | https://huggingface.co/knazarali3/llama3_COMPETITION_MATH |
| llemma   | Meredith  | https://huggingface.co/mrush30/cs6220-llemma_both |

Each branch has either training and inference code for our individual model or separate analysis.

# Datasets
## Processed SAT Math Dataset: train and test splits
sat_math_dict = load_dataset("knazarali3/group2_processed_sat_math_cot")
```
DatasetDict({
    train: Dataset({
        features: ['id', 'question', 'answer', 'is_correct', 'correct_reasoning_chain', 'index'],
        num_rows: 25158
    })
    test: Dataset({
        features: ['id', 'question', 'answer', 'is_correct', 'correct_reasoning_chain', 'index'],
        num_rows: 6290
    })
})
```

## Competition Math Dataset: train and test splits
competition_math_dict = load_dataset("jeggers/competition_math", "original")
```
DatasetDict({
    train: Dataset({
        features: ['problem', 'level', 'type', 'solution', 'extracted_solution'],
        num_rows: 7500
    })
    test: Dataset({
        features: ['problem', 'level', 'type', 'solution', 'extracted_solution'],
        num_rows: 5000
    })
})
```
