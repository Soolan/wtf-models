import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./body_finetuned_model",
    overwrite_output_dir=True,
    num_train_epochs=1,  # You can start with a small number of epochs
    per_device_train_batch_size=1,  # Reduce batch size for small dataset
    save_steps=100,  # Save more frequently for smaller datasets
    save_total_limit=1,  # Limit the number of saved checkpoints
    prediction_loss_only=True,
    evaluation_strategy="steps",
    eval_steps=100,  # Evaluate more frequently
    dataloader_num_workers=4,
    deepspeed="./ds_config.json",  # Optional: Use DeepSpeed for faster training
)

# Prepare the training dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="./training-datasets/body01.csv",  # Replace with the actual file path
    block_size=512,
)

# Initialize the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model on headlines
trainer.train()
trainer.save_model()

