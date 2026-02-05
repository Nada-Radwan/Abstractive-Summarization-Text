# Abstractive Summarization Project

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange)
![Dataset](https://img.shields.io/badge/Dataset-CNN%2FDailyMail-green)

This project implements an **abstractive text summarization** model using **DistilBART** trained on the CNN/DailyMail dataset. The model generates concise summaries of long articles, focusing on capturing the main points while remaining human-readable.

---

## 🚀 Features
- Abstractive summarization (not extractive)  
- Trained on CNN/DailyMail dataset (subset used for fast training)  
- Evaluation using **ROUGE scores**  
- Easy-to-use inference pipeline

---

## 📂 Project Structure
Abstractive-Summarization/
├─ Train.py # Traininng model and tokenizer
├─ summarization_model/ # Trained model and tokenizer
├─ evaluateModel.py # Evaluate model with ROUGE
├─ infer.py # Simple infering example
├─ app.py #FastAPI
├─ tastImpots.py # Certainit
└─ README.md # Project documentation
---

## 🛠 Installation
1. Clone the repository:
```bash
git clone https://github.com/Nada-Radwan/Abstractive-Summarization-Text.git
cd Abstractive-Summarization-Text

