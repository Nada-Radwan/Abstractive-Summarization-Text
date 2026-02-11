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

-  Abstractive summarization (sequence-to-sequence generation)
-  Fine-tuned **DistilBART** model
-  Evaluation using **ROUGE-1, ROUGE-2, ROUGE-L**
-  REST API deployment using **FastAPI + Uvicorn**
-  Modular and clean project structure

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

## 📈 Example Output

**Input:**

> Artificial intelligence has become one of the most transformative technologies of the modern era. Over the past decade, rapid advances in machine learning and deep neural networks have enabled computers to perform tasks that were once considered uniquely human. These tasks include image recognition, natural language understanding, and complex decision-making processes. As a result, artificial intelligence systems are now being integrated into many aspects of daily life. In the healthcare industry, artificial intelligence is being used to assist doctors in diagnosing diseases, analyzing medical images, and predicting patient outcomes. AI-powered systems can process vast amounts of medical data in a fraction of the time required by humans, helping clinicians make more accurate and timely decisions. The impact of artificial intelligence is also evident in the financial sector. Banks and financial institutions rely on AI algorithms to detect fraudulent transactions, assess credit risk, and automate customer support through intelligent chatbots. Despite its many benefits, the widespread adoption of artificial intelligence presents significant challenges. Ethical considerations such as transparency, accountability, and fairness must be addressed to ensure responsible use. Looking ahead, artificial intelligence is expected to play an even greater role in shaping the future of work and society. The key to maximizing its benefits lies in thoughtful implementation and continuous oversight.


**Generated Summary:**

> Artificial intelligence is transforming industries through advances in machine learning and neural networks. It is widely used in healthcare and finance to improve efficiency and decision-making. Despite its benefits, ethical challenges must be addressed to ensure responsible adoption.


---

## 🔮 Future Improvements

- Add beam search optimization
- Improve summary length control
- Add Docker support
- Add model versioning and logging

---

## 👩‍💻 Author

**Nada Radwan**  
AI & Machine Learning Engineer  

GitHub: https://github.com/Nada-Radwan

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

