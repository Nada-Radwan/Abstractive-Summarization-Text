# Abstractive Summarization Project

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange)
![Dataset](https://img.shields.io/badge/Dataset-CNN%2FDailyMail-green)


An end-to-end Abstractive Text Summarization system built using DistilBART and deployed with FastAPI.
The model generates concise summaries of long articles, focusing on capturing the main points while remaining human-readable.

📌 Overview

This project implements an abstractive summarization model trained on the CNN/DailyMail dataset.

Unlike extractive summarization (which copies sentences), this model generates new sentences that capture the key meaning of the article in a human-readable way.

The system includes:

Model training

ROUGE-based evaluation

Inference pipeline

FastAPI deployment for real-time summarization

🚀 Features

✔️ Abstractive summarization (sequence-to-sequence generation)
✔️ Fine-tuned DistilBART model
✔️ Evaluation using ROUGE-1, ROUGE-2, ROUGE-L
✔️ REST API deployment using FastAPI + Uvicorn
✔️ Clean modular project structure

🏗 Model Architecture

Base Model: sshleifer/distilbart-cnn-12-6

Framework: PyTorch

Tokenizer: HuggingFace Tokenizer

Training Dataset: CNN/DailyMail (subset used for faster experimentation)

📂 Project Structure
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

🛠 Installation
1️⃣ Clone Repository
git clone https://github.com/Nada-Radwan/Abstractive-Summarization-Text.git
cd Abstractive-Summarization-Text

2️⃣ Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

🏋️ Training

To fine-tune the model:

python Train.py


The trained model will be saved inside:

summarization_model/

📊 Evaluation

Evaluate using ROUGE:

python evaluateModel.py


Metrics calculated:

ROUGE-1

ROUGE-2

ROUGE-L

🧪 Inference (Local)
python infer.py

🌐 Run as API (Deployment)

Start FastAPI server:

uvicorn app:app --reload


Then open:

http://127.0.0.1:8000/docs


You can test the /summarize endpoint directly from Swagger UI.

Example request body:

{
  "text": "Long article text goes here..."
}

📈 Example Output

Input:

Artificial intelligence has become one of the most transformative technologies...

Output:

Artificial intelligence is transforming industries by enabling machines to perform complex tasks once limited to humans.
