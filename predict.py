from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf
import uvicorn
import os
import sys

# 🔧 경로 설정 (KoBERT 토크나이저를 로드하기 위해)
sys.path.append(".")

from tokenization_kobert import KoBertTokenizer

# 📦 모델 및 토크나이저 경로 (KoBERT/saved_model 안에 있다고 가정)
model_dir = "KoBERT/saved_model"

model = TFAutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
tokenizer = KoBertTokenizer(
    vocab_file=os.path.join(model_dir, "tokenizer_78b3253a26.model"),
    vocab_txt=os.path.join(model_dir, "vocab.txt")
)

# 🚀 FastAPI 인스턴스 생성
app = FastAPI()

# ✅ 루트 경로 응답 추가
@app.get("/")
def root():
    return {"message": "Voice phishing detection API is working!"}

# 📨 입력 텍스트 구조 정의
class InputText(BaseModel):
    text: str

# 🔮 예측 함수
def predict_text(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    confidence = tf.nn.softmax(logits, axis=1).numpy()[0][predicted_class]
    return {
        "prediction": int(predicted_class),
        "confidence": float(confidence)
    }

# 🌐 POST 엔드포인트 정의
@app.post("/predict")
def predict(input: InputText):
    return predict_text(input.text)