import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import speech_recognition as sr
from PIL import Image
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
import os
import tempfile
from fpdf import FPDF  # For PDF generation
from datetime import datetime

# NOTE: Ensure you have Kaleido installed for image export:
#       pip install -U kaleido

# Define the expected CSV columns (always defined)
columns = ["Week", "Cognitive Health", "Physical Activity", "Social Engagement", 
           "Diet & Nutrition", "Mental Well-being", "Sleep Quality", "Stress Levels", 
           "Hydration Level", "Speech Complexity", "Image Analysis", "Assessment Score", 
           "Predicted Score"]

# Define features for scaling (exclude columns like "Week" and "Predicted Score")
features_to_scale = [
    "Cognitive Health", "Physical Activity", "Social Engagement", 
    "Diet & Nutrition", "Mental Well-being", "Sleep Quality", 
    "Stress Levels", "Hydration Level", "Speech Complexity", 
    "Image Analysis", "Assessment Score"
]

# Streamlit Page Config
st.set_page_config(page_title="Dementia Care Suite", page_icon="🧠", layout="wide")

# Initialize session state
if 'assessment_done' not in st.session_state:
    st.session_state.assessment_done = False

# Memory for storing past records
history_file = "dementia_history.csv"
if not os.path.exists(history_file):
    pd.DataFrame(columns=columns).to_csv(history_file, index=False)

# Load historical data
try:
    past_data = pd.read_csv(history_file)
    if past_data.empty:
        st.warning("No historical data found. Starting fresh.")
except pd.errors.EmptyDataError:
    st.warning("The history file is empty. Starting fresh.")
    past_data = pd.DataFrame(columns=columns)

# ======================
# PDF Report Generation
# ======================
def generate_pdf_report(user_data, predicted_score, assessment_score, time_period="Weekly", historical_data=None):
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
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(200, 10, txt="User Input Data:", ln=True)
    pdf.set_font("Arial", size=10)
    for key, value in user_data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    pdf.ln(10)

    # Analysis Results
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(200, 10, txt="Analysis Results:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Predicted Cognitive Health Score: {predicted_score:.2f}/100", ln=True)
    pdf.cell(200, 10, txt=f"Cognitive Assessment Score: {assessment_score}/20", ln=True)
    pdf.ln(10)

    # Recommendations (using plain text)
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(200, 10, txt="Personalized Recommendations:", ln=True)
    pdf.set_font("Arial", size=10)
    if predicted_score > 80:
        pdf.multi_cell(200, 10, txt="Excellent cognitive health! Maintain your routine with:\n- Daily mental exercises\n- Regular social interactions\n- Balanced diet and hydration")
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
        
        # Add a section header page for graphs
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Historical Data Graphs", ln=True)
        pdf.ln(5)
        
        # Iterate over metrics (exclude "Week" and "Predicted Score")
        for col in historical_data.columns[1:-1]:
            # Create a Plotly line chart for the column vs. Week
            fig = px.line(historical_data.sort_values("Week"), x="Week", y=col, title=f"Historical {col} Trends")
            # Update x-axis to show every week (dtick=1)
            fig.update_xaxes(tickmode="linear", dtick=1)
            
            # Save the figure as a temporary PNG file (requires Kaleido)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.close()
            try:
                fig.write_image(tmp.name)
            except Exception as e:
                st.error("Image export failed. Please ensure Kaleido is installed using: pip install -U kaleido")
                return None
            temp_image_files.append(tmp.name)
            
            # Add a new page for each graph
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Historical {col} Trends", ln=True)
            pdf.ln(5)
            # Insert the image; adjust width as needed (here, 190 mm)
            pdf.image(tmp.name, x=10, w=190)
    
    # Save PDF
    pdf_file = f"{time_period.lower()}report{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_file)
    
    # Clean up temporary image files
    for filename in temp_image_files:
        os.remove(filename)
    
    return pdf_file

# ======================
# Image Recognition Module
# ======================
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

# ======================
# Voice Recognition Module
# ======================
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

# ======================
# Cognitive Assessment Test
# ======================
def cognitive_assessment():
    st.subheader("🧠 Cognitive Assessment Test")
    score = 0
    
    with st.form("assessment_form"):
        st.write("Rate the following on a scale of 0 (Never) to 4 (Very Often):")
        q1 = st.slider("How often do you forget recent events?", 0, 4, 0)
        q2 = st.slider("Difficulty following conversations?", 0, 4, 0)
        q3 = st.slider("Trouble finding the right words?", 0, 4, 0)
        q4 = st.slider("Problems with daily tasks?", 0, 4, 0)
        q5 = st.slider("Disorientation in familiar places?", 0, 4, 0)
        
        if st.form_submit_button("Submit Assessment"):
            score = q1 + q2 + q3 + q4 + q5
            st.session_state.assessment_score = score
            st.session_state.assessment_done = True
    
    if st.session_state.assessment_done:
        st.write(f"*Assessment Score:* {st.session_state.assessment_score}/20")
        if st.session_state.assessment_score > 15:
            st.error("High risk detected - Please consult a specialist")
        elif st.session_state.assessment_score > 10:
            st.warning("Moderate risk detected - Monitor closely")
        else:
            st.success("Low risk detected - Maintain healthy habits")
    
    return score if st.session_state.assessment_done else 0

# ======================
# Neural Network Model
# ======================
class DementiaModel(nn.Module):
    def _init_(self):
        super(DementiaModel, self)._init_()
        self.lstm = nn.LSTM(input_size=11, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)

# ======================
# Main Application
# ======================
st.title("🧠 Dementia Care Suite - Comprehensive Monitoring System")

# Sidebar Inputs
st.sidebar.header("Patient Data Input")

# Image Upload
uploaded_image = st.sidebar.file_uploader("Upload Medical Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
image_analysis = ""
if uploaded_image:
    image_analysis = analyze_image(uploaded_image)
    st.sidebar.write(f"*Image Analysis:* {image_analysis}")

# Voice Analysis
audio_file = st.sidebar.file_uploader("Upload Voice Sample (WAV)", type=["wav"])
voice_analysis = {}
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        voice_analysis = analyze_audio(tmp_file.name)
    
    if 'error' not in voice_analysis:
        st.sidebar.write(f"*Speech Analysis:* {voice_analysis['sentiment']} sentiment")
        st.sidebar.write(f"Word Count: {voice_analysis['word_count']}")

# Cognitive Assessment
assessment_score = cognitive_assessment()

# Health Parameters
st.sidebar.header("Weekly Health Metrics")
cognitive_health = st.sidebar.slider("Cognitive Health (0-100)", 0, 100, 75)
physical_activity = st.sidebar.number_input("Steps Per Day", min_value=0, value=5000)
social_engagement = st.sidebar.slider("Social Engagement (0-100)", 0, 100, 60)
diet_nutrition = st.sidebar.slider("Diet & Nutrition (0-100)", 0, 100, 70)
mental_wellbeing = st.sidebar.slider("Mental Well-being (0-100)", 0, 100, 60)
sleep_quality = st.sidebar.slider("Sleep Quality (0-100)", 0, 100, 70)
stress_levels = st.sidebar.slider("Stress Levels (0-100)", 0, 100, 40)
hydration_level = st.sidebar.slider("Hydration Level (0-100)", 0, 100, 80)

# Data Collection
user_data = {
    "Cognitive Health": cognitive_health,
    "Physical Activity": physical_activity,
    "Social Engagement": social_engagement,
    "Diet & Nutrition": diet_nutrition,
    "Mental Well-being": mental_wellbeing,
    "Sleep Quality": sleep_quality,
    "Stress Levels": stress_levels,
    "Hydration Level": hydration_level,
    "Speech Complexity": voice_analysis.get('word_count', 0),
    "Image Analysis": 1 if image_analysis == "Normal" else 0,
    "Assessment Score": assessment_score
}

# Data Preprocessing using the defined features
scaler = MinMaxScaler()
if not past_data.empty and set(features_to_scale).issubset(set(past_data.columns)):
    scaler.fit(past_data[features_to_scale])
    user_data_scaled = scaler.transform(pd.DataFrame([user_data])[features_to_scale])
else:
    user_data_scaled = pd.DataFrame([user_data])[features_to_scale].values

user_data_tensor = torch.tensor(user_data_scaled, dtype=torch.float32).unsqueeze(0)

# Prediction Model
model = DementiaModel()

def predict_risk():
    with torch.no_grad():
        prediction = model(user_data_tensor)
    return prediction.item() * 100

# Generate Report
if st.sidebar.button("Generate Comprehensive Report"):
    if not st.session_state.assessment_done:
        st.warning("Please complete the cognitive assessment first!")
    else:
        predicted_score = predict_risk()
        
        # Save data
        new_entry = pd.DataFrame([user_data])
        new_entry["Predicted Score"] = predicted_score
        new_entry.insert(0, "Week", len(past_data) + 1)
        
        updated_data = pd.concat([past_data, new_entry], ignore_index=True)
        updated_data.to_csv(history_file, index=False)
        past_data = updated_data
        
        # Display results
        st.subheader("📈 Risk Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Cognitive Score", f"{predicted_score:.1f}/100")
        with col2:
            st.metric("Clinical Assessment Score", f"{assessment_score}/20")
        
        # Recommendations
        st.subheader("📋 Personalized Recommendations")
        if predicted_score > 80:
            st.success("""
Excellent cognitive health! Maintain your routine with:
- Daily mental exercises
- Regular social interactions
- Balanced diet and hydration
            """)
        elif predicted_score > 60:
            st.warning("""
Mild cognitive decline detected. Consider:
- Increasing physical activity
- Cognitive training exercises
- Stress management techniques
            """)
        else:
            st.error("""
Significant risk detected. Immediate actions needed:
- Consult a neurologist
- Implement structured daily routine
- Engage in supervised cognitive therapy
            """)
        
        # Generate PDF Report (including historical graphs showing all weeks)
        pdf_file = generate_pdf_report(user_data, predicted_score, assessment_score, "Weekly", historical_data=past_data)
        if pdf_file:
            # Store the last generated PDF in session state
            st.session_state["last_pdf"] = pdf_file
            with open(pdf_file, "rb") as file:
                st.download_button(
                    label="📥 Download Weekly Report (PDF)",
                    data=file,
                    file_name=pdf_file,
                    mime="application/pdf"
                )

# Data Management
st.sidebar.header("Data Management")
if st.sidebar.button("Clear Patient History"):
    open(history_file, 'w').close()
    past_data = pd.DataFrame(columns=columns)
    st.sidebar.warning("History cleared!")
    st.experimental_rerun()

st.sidebar.download_button(
    label="Export Patient Data",
    data=past_data.to_csv(index=False),
    file_name="dementia_care_data.csv",
    mime="text/csv"
)

# Additional Download Button for the Last Generated PDF Report
if "last_pdf" in st.session_state and os.path.exists(st.session_state["last_pdf"]):
    with open(st.session_state["last_pdf"], "rb") as file:
        st.sidebar.download_button(
            label="Download Last Generated Report (PDF)",
            data=file,
            file_name=st.session_state["last_pdf"],
            mime="application/pdf"
        )

# System Information
st.sidebar.markdown("---")
st.sidebar.markdown("*System Info:*")
st.sidebar.markdown(f"- Patients Tracked: {len(past_data)}")
st.sidebar.markdown(f"- Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Historical Data Visualization in the app
if not past_data.empty:
    st.subheader("📊 Historical Data Overview")
    for column in past_data.columns[1:-1]:
        fig = px.line(past_data, x="Week", y=column, title=f"Historical {column} Trends")
        # Update the x-axis to show all weeks (using linear tick mode)
        fig.update_xaxes(tickmode="linear", dtick=1)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("ℹ No historical data available. Generate a report to see visualizations.")

if st.sidebar.button("🗑 Erase All Data"):
    if os.path.exists(history_file):
        os.remove(history_file)
    past_data = pd.DataFrame(columns=columns)
    st.sidebar.warning("⚠ All Stored Data Erased Successfully!")
    st.experimental_rerun()