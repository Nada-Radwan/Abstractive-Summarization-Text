# 🧠 Abstractive Text Summarization with DistilBART

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![Dataset](https://img.shields.io/badge/Dataset-CNN%2FDailyMail-green)
![API](https://img.shields.io/badge/API-FastAPI-teal)

> An end-to-end **Abstractive Text Summarization** system built using **DistilBART**, trained on the **CNN/DailyMail dataset**, and deployed with **FastAPI**.

---

## 📌 Overview

This project implements an **abstractive summarization model** capable of generating concise summaries from long articles.

Unlike extractive summarization (which copies sentences), this model **generates new sentences** that capture the key meaning of the text while maintaining readability and coherence.

The system includes:

- Model fine-tuning
- ROUGE-based evaluation
- Inference pipeline
- FastAPI deployment for real-time summarization

---

## 🚀 Features

- ✔️ Abstractive summarization (sequence-to-sequence generation)
- ✔️ Fine-tuned **DistilBART** model
- ✔️ Evaluation using **ROUGE-1, ROUGE-2, ROUGE-L**
- ✔️ REST API deployment using **FastAPI + Uvicorn**
- ✔️ Modular and clean project structure

---

## 🏗 Model Details

- **Base Model:** `sshleifer/distilbart-cnn-12-6`
- **Framework:** PyTorch
- **Tokenizer:** HuggingFace Tokenizer
- **Training Dataset:** CNN/DailyMail (subset used for faster experimentation)

---

## 📂 Project Structure

```
Abstractive-Summarization-Text/
│
├── Train.py                # Model training script
├── evaluateModel.py        # ROUGE evaluation
├── infer.py                # Local inference example
├── app.py                  # FastAPI deployment
├── testImports.py          # Environment validation
├── summarization_model/    # Saved model & tokenizer
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

---

## 🛠 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Nada-Radwan/Abstractive-Summarization-Text.git
cd Abstractive-Summarization-Text
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

To fine-tune the model:

```bash
python Train.py
```

The trained model will be saved inside:

```
summarization_model/
```

---

## 📊 Evaluation

Evaluate the model using ROUGE metrics:

```bash
python evaluateModel.py
```

Metrics calculated:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum

---

## 🧪 Inference (Local)

Run inference on a sample input:

```bash
python infer.py
```

---

## 🌐 Run as API (Deployment)

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Then open:

```
http://127.0.0.1:8000/docs
```

You can test the `/summarize` endpoint directly from Swagger UI.

### Example Request

```json
{
  "text": "Artificial intelligence is transforming industries worldwide by enabling machines to perform tasks that once required human intelligence."
}
```

---

## 📈 Example Output

**Input:**

> Artificial intelligence has become one of the most transformative technologies of the modern era...

**Generated Summary:**

> Artificial intelligence is transforming industries by enabling machines to perform complex tasks once limited to humans.

---

## 🔮 Future Improvements

- Add beam search optimization
- Improve summary length control
- Add Docker support
- Deploy to Azure or AWS
- Add model versioning and logging

---

## 👩‍💻 Author

**Nada Radwan**  
AI & Machine Learning Engineer  

GitHub: https://github.com/Nada-Radwan

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
