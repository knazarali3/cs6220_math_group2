# cs6220_math_group2

SAT Math Dataset from HuggingFace; We only take questions with correct solutions.
80% Train 10% Test 10% Validation

DatasetDict({ <br/> 
    train: Dataset({ <br/> 
        features: ['id', 'question', 'reasoning_chain', 'answer', 'is_correct'], <br/> 
        num_rows: 29244 <br/> 
    }) <br/> 
    test: Dataset({ <br/> 
        features: ['id', 'question', 'reasoning_chain', 'answer', 'is_correct'], <br/> 
        num_rows: 1625 <br/> 
    }) <br/> 
    valid: Dataset({ <br/> 
        features: ['id', 'question', 'reasoning_chain', 'answer', 'is_correct'], <br/> 
        num_rows: 1625 <br/> 
    }) <br/> 
}) <br/> 
