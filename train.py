# ===============================
# 1. Imports
# ===============================
import torch
import warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

print("Imports loaded successfully")

# ===============================
# 2. Model & Tokenizer
# ===============================
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model and tokenizer loaded")
print(f"Model name: {MODEL_NAME}")
print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# ===============================
# 3. Load Dataset (SUBSET)
# ===============================
print("Loading CNN/DailyMail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Reduce size for fast training
train_data = dataset["train"].shuffle(seed=42).select(range(5000))
val_data = dataset["validation"].shuffle(seed=42).select(range(500))

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# ===============================
# 4. Preprocessing Function
# ===============================
def preprocess(batch):
    inputs = tokenizer(
        batch["article"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

    outputs = tokenizer(
        batch["highlights"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    inputs["labels"] = outputs["input_ids"]
    return inputs

print("Starting preprocessing...")
train_data = train_data.map(
    preprocess,
    batched=True,
    desc="Preprocessing training data"
)

val_data = val_data.map(
    preprocess,
    batched=True,
    desc="Preprocessing validation data"
)

print("Preprocessing completed")

# ===============================
# 5. Data Collator
# ===============================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# ===============================
# 6. Training Arguments
# ===============================
training_args = TrainingArguments(
    output_dir="./summarization_model",
    eval_strategy="steps",     
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=1,             
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# ===============================
# 7. Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator
)

# ===============================
# 8. Train
# ===============================
print("Training started...")
trainer.train()
print("Training finished")

# ===============================
# 9. Save Model
# ===============================
print("Saving model and tokenizer...")
trainer.save_model("./summarization_model")
tokenizer.save_pretrained("./summarization_model")
print("Model saved successfully")
