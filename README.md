# cs6220_math_group2

## Processed SAT Math Dataset: train and test splits
sat_math_dict = load_dataset("knazarali3/group2_processed_sat_math_cot")
print(sat_math_dict)
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
print(competition_math_dict)
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