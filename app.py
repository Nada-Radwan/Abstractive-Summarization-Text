from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI(title="Abstractive Summarization API")

# Load your trained model
MODEL_PATH = "./summarization_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class TextInput(BaseModel):
    text: str

@app.post("/summarize")
def summarize(data: TextInput):
    text = data.text.strip()

    # Truncate if text is too long
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    summary_ids = model.generate(
        **inputs,
        max_length=130,
        min_length=40,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        do_sample=False
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )


    return {"summary": summary}
