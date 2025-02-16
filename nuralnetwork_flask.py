# app.py

from flask import Flask, request, render_template, send_file, redirect, url_for, flash
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.express as px
import speech_recognition as sr
from PIL import Image
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
import os
import tempfile
from fpdf import FPDF
from datetime import datetime

# ============== Configuration ==============
app = Flask(_name_)
app.secret_key = "some_secret_key"  # for flash messages, etc.

# CSV columns
columns = [
    "Week", "Cognitive Health", "Physical Activity", "Social Engagement",
    "Diet & Nutrition", "Mental Well-being", "Sleep Quality", "Stress Levels",
    "Hydration Level", "Speech Complexity", "Image Analysis", "Assessment Score",
    "Predicted Score"
]

# Features for scaling (exclude "Week" and "Predicted Score")
features_to_scale = [
    "Cognitive Health", "Physical Activity", "Social Engagement",
    "Diet & Nutrition", "Mental Well-being", "Sleep Quality",
    "Stress Levels", "Hydration Level", "Speech Complexity",
    "Image Analysis", "Assessment Score"
]

history_file = "dementia_history.csv"
if not os.path.exists(history_file):
    pd.DataFrame(columns=columns).to_csv(history_file, index=False)

# Load historical data
try:
    past_data = pd.read_csv(history_file)
except pd.errors.EmptyDataError:
    past_data = pd.DataFrame(columns=columns)


# ============== Neural Network Model ==============
class DementiaModel(nn.Module):
    def _init_(self):
        super(DementiaModel, self)._init_()
        self.lstm = nn.LSTM(input_size=11, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)

model = DementiaModel()  # A simple LSTM model (untrained in this example)

def predict_risk(user_data_scaled):
    """Run the LSTM model and return a predicted score."""
    user_data_tensor = torch.tensor(user_data_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(user_data_tensor)
    return prediction.item() * 100


# ============== Utility Functions ==============
def analyze_image(image_path):
    """Analyze the uploaded image using a pretrained ResNet18 model."""
    model_resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model_resnet.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model_resnet(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return "Normal" if probabilities[0] > 0.5 else "Abnormal"

def analyze_audio(audio_path):
    """Analyze the uploaded audio for sentiment & word count."""
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
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

def generate_pdf_report(user_data, predicted_score, assessment_score, time_period="Weekly", historical_data=None):
    """Generate a PDF report including textual analysis and historical graphs."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Report Header
    pdf.cell(200, 10, txt=f"Dementia Care Suite {time_period} Report", ln=True, align="C")
    pdf.ln(10)

    # Report Date
    pdf.cell(200, 10, txt=f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)

    # User Data
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="User Input Data:", ln=True)
    pdf.set_font("Arial", size=10)
    for key, value in user_data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.ln(10)

    # Analysis Results
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Analysis Results:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Predicted Cognitive Health Score: {predicted_score:.2f}/100", ln=True)
    pdf.cell(200, 10, txt=f"Cognitive Assessment Score: {assessment_score}/20", ln=True)
    pdf.ln(10)

    # Recommendations
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Personalized Recommendations:", ln=True)
    pdf.set_font("Arial", size=10)
    if predicted_score > 80:
        pdf.multi_cell(200, 10, txt="Excellent cognitive health! Maintain your routine:\n- Daily mental exercises\n- Regular social interactions\n- Balanced diet and hydration")
    elif predicted_score > 60:
        pdf.multi_cell(200, 10, txt="Mild cognitive decline detected. Consider:\n- Increasing physical activity\n- Cognitive training exercises\n- Stress management techniques")
    else:
        pdf.multi_cell(200, 10, txt="Significant risk detected. Immediate actions needed:\n- Consult a neurologist\n- Implement structured daily routine\n- Engage in supervised cognitive therapy")
    pdf.ln(10)

    # Historical Graphs Section
    temp_image_files = []
    if historical_data is not None and not historical_data.empty:
        # Ensure the Week column is numeric
        historical_data["Week"] = pd.to_numeric(historical_data["Week"], errors='coerce')
        
        # Add a new page for graphs
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Historical Data Graphs", ln=True)
        pdf.ln(5)
        
        # Iterate over metrics (exclude "Week" and "Predicted Score")
        for col in historical_data.columns[1:-1]:
            fig = px.line(historical_data.sort_values("Week"), x="Week", y=col, title=f"Historical {col} Trends")
            fig.update_xaxes(tickmode="linear", dtick=1)
            
            # Save the figure as a temporary PNG file (requires Kaleido)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.close()
            try:
                fig.write_image(tmp.name)
            except Exception as e:
                # If image export fails, skip
                continue
            temp_image_files.append(tmp.name)
            
            # Add a new page for each graph
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Historical {col} Trends", ln=True)
            pdf.ln(5)
            pdf.image(tmp.name, x=10, w=190)

    pdf_file = f"{time_period.lower()}report{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_file)
    
    # Clean up temporary image files
    for filename in temp_image_files:
        os.remove(filename)
    
    return pdf_file


# ============== Flask Routes ==============

@app.route("/", methods=["GET", "POST"])
def index():
    """Main page with forms to upload data, fill metrics, and do the cognitive assessment."""
    global past_data
    
    if request.method == "POST":
        # 1. Handle Image Upload
        image_file = request.files.get("medical_image")
        image_analysis_result = ""
        if image_file and image_file.filename != "":
            with tempfile.NamedTemporaryFile(delete=False) as tmp_img:
                image_file.save(tmp_img.name)
                image_analysis_result = analyze_image(tmp_img.name)

        # 2. Handle Audio Upload
        audio_file = request.files.get("voice_sample")
        voice_analysis_result = {}
        if audio_file and audio_file.filename != "":
            with tempfile.NamedTemporaryFile(delete=False) as tmp_audio:
                audio_file.save(tmp_audio.name)
                voice_analysis_result = analyze_audio(tmp_audio.name)
        
        # 3. Cognitive Assessment
        # We'll read from form fields named q1, q2, etc.
        q1 = int(request.form.get("q1", 0))
        q2 = int(request.form.get("q2", 0))
        q3 = int(request.form.get("q3", 0))
        q4 = int(request.form.get("q4", 0))
        q5 = int(request.form.get("q5", 0))
        assessment_score = q1 + q2 + q3 + q4 + q5

        # 4. Weekly Health Metrics
        try:
            cognitive_health = float(request.form.get("cognitive_health", 0))
            physical_activity = float(request.form.get("physical_activity", 0))
            social_engagement = float(request.form.get("social_engagement", 0))
            diet_nutrition = float(request.form.get("diet_nutrition", 0))
            mental_wellbeing = float(request.form.get("mental_wellbeing", 0))
            sleep_quality = float(request.form.get("sleep_quality", 0))
            stress_levels = float(request.form.get("stress_levels", 0))
            hydration_level = float(request.form.get("hydration_level", 0))
        except ValueError:
            flash("Invalid numeric input for health metrics.", "danger")
            return redirect(url_for("index"))

        # 5. Create user_data dictionary
        user_data = {
            "Cognitive Health": cognitive_health,
            "Physical Activity": physical_activity,
            "Social Engagement": social_engagement,
            "Diet & Nutrition": diet_nutrition,
            "Mental Well-being": mental_wellbeing,
            "Sleep Quality": sleep_quality,
            "Stress Levels": stress_levels,
            "Hydration Level": hydration_level,
            "Speech Complexity": voice_analysis_result.get("word_count", 0),
            "Image Analysis": 1 if image_analysis_result == "Normal" else 0,
            "Assessment Score": assessment_score
        }

        # 6. Scale the data
        scaler = MinMaxScaler()
        if not past_data.empty and set(features_to_scale).issubset(set(past_data.columns)):
            scaler.fit(past_data[features_to_scale])
            user_data_scaled = scaler.transform(pd.DataFrame([user_data])[features_to_scale])
        else:
            user_data_scaled = pd.DataFrame([user_data])[features_to_scale].values
        
        # 7. Predict risk
        predicted_score = predict_risk(user_data_scaled)

        # 8. Save new data row
        new_entry = pd.DataFrame([user_data])
        new_entry["Predicted Score"] = predicted_score
        new_entry.insert(0, "Week", len(past_data) + 1)
        past_data = pd.concat([past_data, new_entry], ignore_index=True)
        past_data.to_csv(history_file, index=False)

        # 9. Generate PDF
        pdf_file = generate_pdf_report(
            user_data,
            predicted_score,
            assessment_score,
            time_period="Weekly",
            historical_data=past_data
        )

        # Prepare data for results page
        # We can store the file path in the session or pass via redirect
        request.session_pdf_path = pdf_file  # In a real app, use a session store
        return render_template("results.html",
            predicted_score=predicted_score,
            assessment_score=assessment_score,
            image_analysis_result=image_analysis_result,
            voice_text=voice_analysis_result.get("text", ""),
            voice_sentiment=voice_analysis_result.get("sentiment", ""),
            pdf_path=pdf_file
        )

    # If GET, show the form
    return render_template("index.html")


@app.route("/download_pdf/<path:pdf_filename>")
def download_pdf(pdf_filename):
    """Download the generated PDF."""
    if not os.path.exists(pdf_filename):
        flash("File not found.", "danger")
        return redirect(url_for("index"))
    return send_file(pdf_filename, as_attachment=True)


@app.route("/download_csv")
def download_csv():
    """Download the entire patient history CSV."""
    if not os.path.exists(history_file):
        flash("No history file found.", "danger")
        return redirect(url_for("index"))
    return send_file(history_file, as_attachment=True)


@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear the patient history CSV."""
    global past_data
    open(history_file, 'w').close()
    past_data = pd.DataFrame(columns=columns)
    flash("History cleared!", "success")
    return redirect(url_for("index"))


@app.route("/erase_all_data", methods=["POST"])
def erase_all_data():
    """Erase all data (history file)."""
    global past_data
    if os.path.exists(history_file):
        os.remove(history_file)
    past_data = pd.DataFrame(columns=columns)
    flash("All stored data erased successfully!", "warning")
    return redirect(url_for("index"))


# ============== Template In-Memory Examples ==============
# For a real application, create 'templates/index.html' and 'templates/results.html' files.

@app.route("/sample_templates")
def sample_templates():
    """
    This route just returns example HTML that you'd place in:
      - templates/index.html
      - templates/results.html
    so you know how to build your own forms.
    """
    index_html = """
<h1>Dementia Care Suite - Flask Example</h1>
<form method="POST" enctype="multipart/form-data">
  <h3>Upload Files</h3>
  Medical Image: <input type="file" name="medical_image"><br>
  Voice Sample (WAV): <input type="file" name="voice_sample"><br><br>

  <h3>Weekly Health Metrics</h3>
  Cognitive Health (0-100): <input type="number" name="cognitive_health" value="75"><br>
  Steps Per Day: <input type="number" name="physical_activity" value="5000"><br>
  Social Engagement (0-100): <input type="number" name="social_engagement" value="60"><br>
  Diet & Nutrition (0-100): <input type="number" name="diet_nutrition" value="70"><br>
  Mental Well-being (0-100): <input type="number" name="mental_wellbeing" value="60"><br>
  Sleep Quality (0-100): <input type="number" name="sleep_quality" value="70"><br>
  Stress Levels (0-100): <input type="number" name="stress_levels" value="40"><br>
  Hydration Level (0-100): <input type="number" name="hydration_level" value="80"><br><br>

  <h3>Cognitive Assessment</h3>
  Rate from 0 (Never) to 4 (Very Often):<br>
  1) How often do you forget recent events? <input type="number" name="q1" value="0"><br>
  2) Difficulty following conversations? <input type="number" name="q2" value="0"><br>
  3) Trouble finding the right words? <input type="number" name="q3" value="0"><br>
  4) Problems with daily tasks? <input type="number" name="q4" value="0"><br>
  5) Disorientation in familiar places? <input type="number" name="q5" value="0"><br><br>

  <button type="submit">Generate Comprehensive Report</button>
</form>
<br>
<form method="POST" action="/clear_history">
  <button type="submit">Clear Patient History</button>
</form>
<form method="POST" action="/erase_all_data">
  <button type="submit">Erase All Data</button>
</form>
<a href="/download_csv">Download Patient Data (CSV)</a>
"""

    results_html = """
<h1>Results Page</h1>
<p>Here you can show the predicted_score, assessment_score, image_analysis, etc.</p>
<p><a href="/download_pdf/YOUR_PDF_FILENAME.pdf">Download the PDF</a></p>
"""

    return f"<h2>index.html</h2><pre>{index_html}</pre><hr><h2>results.html</h2><pre>{results_html}</pre>"


# ============== Main ==============
if _name_ == "_main_":
    # You can run python app.py and go to http://127.0.0.1:5000
    app.run(debug=True)