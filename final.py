import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import os

# Streamlit Page Config
st.set_page_config(page_title="Dementia Reduction Tracker", page_icon="🧠", layout="wide")

# Memory for storing past records
history_file = "dementia_history.csv"
if not os.path.exists(history_file):
    pd.DataFrame(columns=["Week", "Cognitive Health", "Physical Activity", "Social Engagement", "Diet & Nutrition", "Mental Well-being", "Sleep Quality", "Stress Levels", "Hydration Level", "Predicted Score"]).to_csv(history_file, index=False)

# User Input for Key Parameters
st.sidebar.header("🧠 Enter Your Weekly Activity Data")
cognitive_health = st.sidebar.slider("Cognitive Health (0-100)", 0, 100, 75)
physical_activity = st.sidebar.number_input("Steps Per Day", min_value=0, value=5000)
social_engagement = st.sidebar.slider("Social Engagement (Conversations per Day)", 0, 10, 5)
diet_nutrition = st.sidebar.slider("Diet & Nutrition (Healthy Eating %)", 0, 100, 70)
mental_wellbeing = st.sidebar.slider("Mental Well-being (0-100)", 0, 100, 60)
sleep_quality = st.sidebar.slider("Sleep Quality (Hours per Night)", 0, 12, 7)
stress_levels = st.sidebar.slider("Stress Levels (0-100)", 0, 100, 40)
hydration_level = st.sidebar.slider("Hydration Level (% of daily requirement)", 0, 100, 80)

# Data Collection
user_data = pd.DataFrame({
    "Cognitive Health": [cognitive_health],
    "Physical Activity": [physical_activity],
    "Social Engagement": [social_engagement],
    "Diet & Nutrition": [diet_nutrition],
    "Mental Well-being": [mental_wellbeing],
    "Sleep Quality": [sleep_quality],
    "Stress Levels": [stress_levels],
    "Hydration Level": [hydration_level]
})

# Normalize Data
scaler = MinMaxScaler()
user_data_scaled = scaler.fit_transform(user_data)
user_data_tensor = torch.tensor(user_data_scaled, dtype=torch.float32).unsqueeze(0)

# Neural Network Model for Prediction
class DementiaModel(nn.Module):
    def __init__(self):
        super(DementiaModel, self).__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)

# Load Model & Predict
model = DementiaModel()
def predict_cognitive_health():
    with torch.no_grad():
        return model(user_data_tensor).item()

if st.sidebar.button("Generate Report"):
    predicted_score = predict_cognitive_health()
    st.title("🧠 Dementia Risk Analysis Report")
    st.write(f"**Predicted Cognitive Health Score:** {predicted_score:.2f}/100")
    
    if predicted_score > 80:
        st.success("✔ Great progress! Maintain your routine.")
    elif 60 <= predicted_score <= 80:
        st.warning("⚠️ Good, but improve social engagement and physical activity.")
    else:
        st.error("❌ Risk of dementia increasing! Increase cognitive exercises and diet quality.")
    
    # Save data to history
    past_data = pd.read_csv(history_file)
    new_entry = user_data.copy()
    new_entry["Predicted Score"] = predicted_score
    new_entry.insert(0, "Week", len(past_data) + 1)
    past_data = pd.concat([past_data, new_entry], ignore_index=True)
    past_data.to_csv(history_file, index=False)
    
    # Performance Comparison Graph
    st.subheader("📊 Performance Comparison Over Time")
    fig_perf = px.line(past_data, x="Week", y="Predicted Score", title="Cognitive Health Trend")
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Visualization of Input Data
    st.subheader("📊 Your Activity Data")
    for column in user_data.columns:
        fig = px.line(past_data, x="Week", y=column, title=f"{column} Progress Over Time")
        st.plotly_chart(fig, use_container_width=True, key=f"{column}_chart")

    # Clear History Button
    if st.sidebar.button("🗑 Clear History"):
        open(history_file, 'w').close()
        st.sidebar.warning("⚠️ History Cleared Successfully!")

        st.sidebar.download_button(
        label="📥 Download History",
        data=pd.read_csv(history_file).to_csv(index=False),
        file_name="dementia_history.csv",
        mime="text/csv"
    )

    st.sidebar.success("✅ Report Generated Successfully!")

