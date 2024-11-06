from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, load_from_disk

# Load the Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
model = AutoModelForCausalLM.from_pretrained("openai-community/openai-gpt")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Assuming the dataset is already loaded as a DatasetDict named `dataset`
dataset = load_from_disk('sat-math-datasets-splits')



# Define a function to preprocess the dataset
def preprocess_function(examples):
    # Concatenate question, reasoning chain, and answer into a single text string
    text = [
        f"Question: {q} Reasoning: {r} Answer: {a}"
        for q, r, a in zip(examples["question"], examples["reasoning_chain"], examples["answer"])
    ]
    # Tokenize the text
    encoding = tokenizer(text, padding="longest", truncation=True, max_length=None)
    # Set labels the same as input_ids for causal language modeling
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

# Apply preprocessing to the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)


# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=1e-5,             
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16,   
    num_train_epochs=2,             
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # Validation split for evaluation
)

# Start Training
trainer.train()

# Evaluate and Save the Model
trainer.evaluate(eval_dataset=tokenized_datasets["valid"])
trainer.save_model("gpt/model")
