# 📊 Sentiment Analysis Fine-Tuning with IndoBERT

## 📖 Project Overview
This repository contains the solution for the Data Scientist Intern Technical Test. The objective of this project is to build and fine-tune a Natural Language Processing (NLP) model to accurately classify sentiment from text data. 

## 📂 Dataset Source & Annotation
* **Source:** https://www.kaggle.com/datasets/jocelyndumlao/indonesia-presidential-candidates-dataset-2024 
* **Annotation:** The dataset is annotated into specific sentiment categories: `Positive,Negative`. 
* **Preprocessing:** The raw text was preprocessed to remove noise (URLs, mentions, special characters) and tokenized to ensure optimal model performance.

## 🧠 Chosen Model
**Model:** `IndoBERT` (Pre-trained Indonesian Language Model via Hugging Face Transformers).
* **Reasoning:** IndoBERT is specifically pre-trained on a massive Indonesian corpus, making it highly capable of understanding the nuances, context, and semantic structures of the Indonesian language compared to standard multilingual models. It was fine-tuned for sequence classification to map the text inputs to their respective sentiment labels accurately.

## 🗺️ System Flowchart
A detailed flowchart illustrating the end-to-end data pipeline—from raw data ingestion and text preprocessing to model training and evaluation—is included in this repository.
* 📄 **View Flowchart:** Please refer to the `flowchart.png` file attached in this submission.

## 🚀 Setup Instructions

### Prerequisites
Make sure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.

### 1. Clone or Extract the Repository
Navigate to the project folder containing the `.ipynb` file and the dataset.

### 2. Create and Activate Virtual Environment (Optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate 

