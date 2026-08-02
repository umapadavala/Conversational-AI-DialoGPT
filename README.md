# 💬 Conversational AI Agent using DialoGPT

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

A context-aware conversational AI built using Microsoft's **DialoGPT-medium** model from Hugging Face Transformers. The chatbot maintains conversation history to generate coherent, contextually relevant responses during multi-turn conversations.

---

## 📌 Overview

Traditional rule-based chatbots often fail to understand context across multiple interactions. This project leverages a pretrained Transformer language model capable of maintaining conversational history and generating natural responses.

---

## ✨ Features

- Multi-turn conversation support
- Context-aware responses
- Hugging Face Transformers integration
- GPU acceleration with PyTorch
- Top-k and Top-p sampling
- Temperature-controlled response generation
- Repetition avoidance using n-gram blocking

---

## ⚙️ Working Pipeline

```
User Input
      │
      ▼
Tokenizer (BPE)
      │
      ▼
Conversation History
      │
      ▼
DialoGPT-medium
      │
      ▼
Sampling Strategy
(Top-k + Top-p + Temperature)
      │
      ▼
Generated Response
```

---

## 🔧 Decoding Strategy

- Model: **microsoft/DialoGPT-medium**
- Tokenization: Byte Pair Encoding (BPE)
- Maximum Context Length: 512 tokens
- Top-k Sampling: 50
- Top-p Sampling: 0.92
- Temperature: 0.75
- No Repeat N-Gram Size: 3

---

## 📂 Project Structure

```
Conversational-AI-DialoGPT/

├── chatbot.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository.

```bash
git clone https://github.com/your-username/Conversational-AI-DialoGPT.git

cd Conversational-AI-DialoGPT
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Chatbot

```bash
python chatbot.py
```

---

## 💬 Sample Conversation

```
User : Hello!

Bot  : Hi! How can I help you today?

User : Can you help me prepare for my Deep Learning exam?

Bot  : Certainly! Let's start with the fundamentals of neural networks.
```

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- DialoGPT-medium
- Tokenizers

---

## 📜 License

This project is licensed under the MIT License.
