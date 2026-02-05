import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

MODEL_PATH = "./summarization_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

rouge = evaluate.load("rouge")

dataset = load_dataset("cnn_dailymail", "3.0.0")["test"].select(range(100))

predictions = []
references = []

for sample in dataset:
    inputs = tokenizer(
        sample["article"],
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    summary_ids = model.generate(
        **inputs,
        max_length=130,
        min_length=30,
        do_sample=False
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    predictions.append(summary)
    references.append(sample["highlights"])

results = rouge.compute(
    predictions=predictions,
    references=references
)

print(results)
