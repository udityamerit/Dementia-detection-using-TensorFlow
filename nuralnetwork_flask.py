from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
import os
import tempfile
import speech_recognition as sr

app = Flask(__name__)

# Ensure the file exists and has the correct columns
history_file = "dementia_history.csv"
if not os.path.exists(history_file):
    columns = ["Week", "Cognitive Health", "Physical Activity", "Social Engagement", 
               "Diet & Nutrition", "Mental Well-being", "Sleep Quality", "Stress Levels", 
               "Hydration Level", "Speech Complexity", "Image Analysis", "Assessment Score", 
               "Predicted Score"]
    pd.DataFrame(columns=columns).to_csv(history_file, index=False)

# Load historical data
try:
    past_data = pd.read_csv(history_file)
    if past_data.empty:
        past_data = pd.DataFrame(columns=columns)
except pd.errors.EmptyDataError:
    past_data = pd.DataFrame(columns=columns)

class DementiaModel(nn.Module):
    def __init__(self):
        super(DementiaModel, self).__init__()
        self.lstm = nn.LSTM(input_size=11, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)

model = DementiaModel()

def analyze_image(uploaded_image):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(uploaded_image).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return "Normal" if probabilities[0] > 0.5 else "Abnormal"

def analyze_audio(audio_file):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            
            sentiment_analyzer = pipeline("sentiment-analysis")
            sentiment = sentiment_analyzer(text)[0]
            
            return {
                "text": text,
                "sentiment": sentiment['label'],
                "confidence": sentiment['score'],
                "word_count": len(text.split())
            }
    except Exception as e:
        return {"error": str(e)}

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        result = analyze_image(file)
        return jsonify({"result": result})

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(file.read())
            result = analyze_audio(tmp_file.name)
        return jsonify(result)

@app.route('/cognitive_assessment', methods=['POST'])
def cognitive_assessment():
    data = request.json
    score = sum(data.values())
    return jsonify({"assessment_score": score})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_data = pd.DataFrame(data, index=[0])
    
    scaler = MinMaxScaler()
    if not past_data.empty:
        scaler.fit(past_data.iloc[:, 1:-1])
    
    if not past_data.empty:
        user_data_scaled = scaler.transform(user_data)
    else:
        user_data_scaled = user_data.values

    user_data_tensor = torch.tensor(user_data_scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(user_data_tensor)
    predicted_score = prediction.item() * 100
    
    return jsonify({"predicted_score": predicted_score})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    user_data = pd.DataFrame(data, index=[0])
    
    scaler = MinMaxScaler()
    if not past_data.empty:
        scaler.fit(past_data.iloc[:, 1:-1])
    
    if not past_data.empty:
        user_data_scaled = scaler.transform(user_data)
    else:
        user_data_scaled = user_data.values

    user_data_tensor = torch.tensor(user_data_scaled, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(user_data_tensor)
    predicted_score = prediction.item() * 100
    
    new_entry = user_data.copy()
    new_entry["Predicted Score"] = predicted_score
    new_entry.insert(0, "Week", len(past_data) + 1)
    
    updated_data = pd.concat([past_data, new_entry], ignore_index=True)
    updated_data.to_csv(history_file, index=False)
    
    return jsonify({"predicted_score": predicted_score, "assessment_score": data["Assessment Score"]})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    open(history_file, 'w').close()
    global past_data
    past_data = pd.DataFrame(columns=past_data.columns)
    return jsonify({"message": "History cleared!"})

@app.route('/export_data', methods=['GET'])
def export_data():
    return past_data.to_csv(index=False)

@app.route('/system_info', methods=['GET'])
def system_info():
    return jsonify({
        "patients_tracked": len(past_data),
        "last_update": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    })

if __name__ == '__main__':
    app.run(debug=True)