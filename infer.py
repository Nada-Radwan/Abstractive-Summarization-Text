import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ===============================
# 1. Load model & tokenizer
# ===============================
MODEL_PATH = "./summarization_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded on {device}")

# ===============================
# 2. Input text
# ===============================
text = """
Artificial intelligence is revolutionizing many industries
by enabling machines to perform tasks that require human intelligence.
"""

# ===============================
# 3. Tokenize
# ===============================
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(device)

# ===============================
# 4. Generate summary
# ===============================
with torch.no_grad():
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=130,
        min_length=30,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

# ===============================
# 5. Decode
# ===============================
summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("\nSummary:")
print(summary)
