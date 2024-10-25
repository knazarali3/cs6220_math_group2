# cs6220_math_group2

SAT Math Dataset from HuggingFace; We only take questions with correct solutions.
80% Train 10% Test 10% Validation

To load dataset,
```
from datasets import load_from_disk

sat_math_datasets_splits = load_from_disk('sat-math-datasets-splits')
print(sat_math_datasets_splits)
```

```
DatasetDict({
    train: Dataset({
        features: ['id', 'question', 'reasoning_chain', 'answer', 'is_correct'], 
        num_rows: 29244
    }) 
    test: Dataset({ 
        features: ['id', 'question', 'reasoning_chain', 'answer', 'is_correct'],
        num_rows: 1625
    }) 
    valid: Dataset({
        features: ['id', 'question', 'reasoning_chain', 'answer', 'is_correct'],  
        num_rows: 1625
    }) 
})
```
