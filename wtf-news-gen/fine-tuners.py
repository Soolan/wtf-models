import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium" or other variants if needed
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to load and prepare data from CSV
def load_csv_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file.readlines()[1:]:  # Skip the header line
            original, optimized = line.strip().split(",")
            data.append((original, optimized))
    return data

# Load data for headings and bodies
headings_data = load_csv_data("headings.csv")
bodies_data = load_csv_data("bodies.csv")

# Define training arguments
training_args_headings = TrainingArguments(
    output_dir="./wtf-news-gen-headings",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

training_args_bodies = TrainingArguments(
    output_dir="./wtf-news-gen-bodies",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

# Function to create datasets and data collators
def create_dataset_and_data_collator(data):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=None,  # We'll provide data directly
        text=data,
        block_size=128  # Adjust as needed
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    return dataset, data_collator

# Create datasets and data collators for headings and bodies
headings_dataset, headings_data_collator = create_dataset_and_data_collator(headings_data)
bodies_dataset, bodies_data_collator = create_dataset_and_data_collator(bodies_data)

# Create trainers and perform fine-tuning for headings and bodies
trainer_headings = Trainer(
    model=model,
    args=training_args_headings,
    data_collator=headings_data_collator,
    train_dataset=headings_dataset,
)

trainer_bodies = Trainer(
    model=model,
    args=training_args_bodies,
    data_collator=bodies_data_collator,
    train_dataset=bodies_dataset,
)

# Fine-tune the models
trainer_headings.train()
trainer_headings.save_model()  # Save the fine-tuned headings model

trainer_bodies.train()
trainer_bodies.save_model()  # Save the fine-tuned bodies model

