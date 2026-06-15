import streamlit as st
import torch
import numpy as np
import re
import warnings
from transformers import AutoTokenizer, AutoModel
from torch import nn

warnings.filterwarnings('ignore')

# --- Konfigurasi Global (Sesuai Notebook) ---
MAX_LEN     = 128
MODEL_NAME  = 'indobenchmark/indobert-base-p1'
MODEL_PATH  = '/content/best_indobert_model.pt'
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Kelas Model (Sesuai Notebook) ---
class IndoBERTClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.5):
        super(IndoBERTClassifier, self).__init__()
        self.bert   = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# --- Helper Function untuk Membersihkan Teks (Sesuai Notebook) ---
def clean_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Memuat Model dan Tokenizer (menggunakan cache Streamlit) ---
@st.cache_resource
def load_model_components():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = IndoBERTClassifier(MODEL_NAME)
    model.to(device)
    
    # Memuat checkpoint model
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval() # Set model ke mode evaluasi
    
    best_threshold = checkpoint['threshold']
    return tokenizer, model, best_threshold

tokenizer, model, best_threshold = load_model_components()

# --- Streamlit UI ---
st.set_page_config(page_title="Analisis Sentimen Tweet", layout="wide")

# Sidebar
st.sidebar.title("Informasi Model")
st.sidebar.write("**Model**: Fine-tuned IndoBERT")
st.sidebar.write("**Akurasi**: 83.80%") # Dari hasil evaluasi terakhir
st.sidebar.write("**Macro F1-Score**: 80.70%") # Dari hasil evaluasi terakhir

st.sidebar.markdown("""
**Parameter Pelatihan Terakhir:**
- Epochs: 8
- Learning Rate: 2e-5
- Patience: 3
- Full Fine-tuning
- Weighted Random Sampler & Weighted CrossEntropy Loss
- Threshold Tuning
""")

st.sidebar.info("Aplikasi ini menganalisis sentimen tweet berbahasa Indonesia (Positif/Negatif) menggunakan model IndoBERT yang sudah dilatih.")

# Main Content
st.title("🇮🇩 Analisis Sentimen Tweet Calon Presiden")
st.markdown("Masukkan tweet dalam bahasa Indonesia di bawah ini untuk menganalisis sentimennya.")

user_input = st.text_area(
    "", 
    height=150, 
    placeholder="Contoh: Prabowo Subianto adalah pilihan terbaik untuk Indonesia!"
)

if st.button("Analisis Sentimen"): # Ubah teks tombol
    if not user_input.strip():
        st.warning("Mohon masukkan teks tweet untuk dianalisis.")
    else:
        cleaned_input = clean_tweet(user_input)
        if not cleaned_input:
            st.warning("Teks yang dimasukkan tidak valid setelah dibersihkan.")
        else:
            # Tokenisasi input
            encoding = tokenizer(
                cleaned_input,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1) # Probabilitas untuk kedua kelas
                positive_prob = probs[:, 1].item() # Probabilitas kelas Positif
            
            # Penentuan sentimen berdasarkan threshold
            prediction = 1 if positive_prob >= best_threshold else 0
            
            # Tampilan hasil
            if prediction == 1:
                st.success(f"### Sentimen: Positif 😊")
                st.write(f"**Keyakinan**: {positive_prob * 100:.2f}%")
                st.write(f"*Threshold yang digunakan: {best_threshold:.2f}*")
            else:
                st.error(f"### Sentimen: Negatif 😠")
                st.write(f"**Keyakinan**: {positive_prob * 100:.2f}%")
                st.write(f"*Threshold yang digunakan: {best_threshold:.2f}*")
            
            st.markdown("---")
            st.info("Teks asli (setelah dibersihkan): " + cleaned_input)

