# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # 1. User Input Collection
# def get_user_input():
#     age = int(input("Enter Age: "))
#     mmse_score = int(input("Enter MMSE Score (0-30): "))
#     mri_brain_volume = float(input("Enter MRI Brain Volume (in cubic mm): "))
#     memory_loss_score = int(input("Enter Memory Loss Score (1-10): "))
#     family_history = int(input("Family History of Dementia? (0 = No, 1 = Yes): "))
#     assessment_difficulty = int(input("Enter Assessment Difficulty (1-5): "))
#     speech_pause_rate = float(input("Enter Speech Pause Rate (words per minute): "))
#     gait_speed = float(input("Enter Gait Speed (meters per second): "))
#     sleep_hours_per_day = float(input("Enter Average Sleep Hours Per Day: "))
#     eye_fixation_time = float(input("Enter Eye Fixation Time (milliseconds): "))
#     hrv_index = float(input("Enter Heart Rate Variability Index (ms): "))
    
#     return pd.DataFrame([[age, mmse_score, mri_brain_volume, memory_loss_score, family_history, assessment_difficulty, speech_pause_rate, gait_speed, sleep_hours_per_day, eye_fixation_time, hrv_index]], 
#                         columns=['Age', 'MMSE_Score', 'MRI_Brain_Volume', 'Memory_Loss_Score', 'Family_History', 'Assessment_Difficulty', 'Speech_Pause_Rate', 'Gait_Speed', 'Sleep_Hours_Per_Day', 'Eye_Fixation_Time', 'HRV_Index'])

# # 2. Sample Dataset for Model Training
# data = {
#     'Age': np.random.randint(50, 90, 200),
#     'MMSE_Score': np.random.randint(10, 30, 200),
#     'MRI_Brain_Volume': np.random.uniform(500, 1500, 200),
#     'Memory_Loss_Score': np.random.randint(1, 10, 200),
#     'Family_History': np.random.choice([0, 1], 200),
#     'Assessment_Difficulty': np.random.randint(1, 5, 200),
#     'Speech_Pause_Rate': np.random.uniform(0.5, 5.0, 200),
#     'Gait_Speed': np.random.uniform(0.1, 1.5, 200),
#     'Sleep_Hours_Per_Day': np.random.uniform(3, 10, 200),
#     'Eye_Fixation_Time': np.random.uniform(100, 800, 200),
#     'HRV_Index': np.random.uniform(20, 100, 200),
#     'Dementia_Level': np.random.choice(['Mild', 'Moderate', 'Severe'], 200)
# }

# df = pd.DataFrame(data)

# # 3. Encoding categorical target variable
# df['Dementia_Level'] = df['Dementia_Level'].map({'Mild': 0, 'Moderate': 1, 'Severe': 2})

# # 4. Splitting dataset
# X = df.drop(columns=['Dementia_Level'])
# y = df['Dementia_Level']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 5. Model Training
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # 6. User Input Prediction
# user_data = get_user_input()
# user_prediction = model.predict(user_data)
# prediction_labels = {0: 'Mild Dementia', 1: 'Moderate Dementia', 2: 'Severe Dementia'}

# print("Predicted Dementia Level:", prediction_labels[user_prediction[0]])

# # 7. Evaluation
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))


# version 2.0

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# 1. User Input Collection
def get_user_input():
    age = int(input("Enter Age: "))
    mmse_score = int(input("Enter MMSE Score (0-30): "))
    mri_brain_volume = float(input("Enter MRI Brain Volume (in cubic mm): "))
    memory_loss_score = int(input("Enter Memory Loss Score (1-10): "))
    family_history = int(input("Family History of Dementia? (0 = No, 1 = Yes): "))
    assessment_difficulty = int(input("Enter Assessment Difficulty (1-5): "))
    speech_pause_rate = float(input("Enter Speech Pause Rate (words per minute): "))
    gait_speed = float(input("Enter Gait Speed (meters per second): "))
    sleep_hours_per_day = float(input("Enter Average Sleep Hours Per Day: "))
    eye_fixation_time = float(input("Enter Eye Fixation Time (milliseconds): "))
    hrv_index = float(input("Enter Heart Rate Variability Index (ms): "))
    
    return pd.DataFrame([[age, mmse_score, mri_brain_volume, memory_loss_score, family_history, assessment_difficulty, speech_pause_rate, gait_speed, sleep_hours_per_day, eye_fixation_time, hrv_index]], 
                        columns=['Age', 'MMSE_Score', 'MRI_Brain_Volume', 'Memory_Loss_Score', 'Family_History', 'Assessment_Difficulty', 'Speech_Pause_Rate', 'Gait_Speed', 'Sleep_Hours_Per_Day', 'Eye_Fixation_Time', 'HRV_Index'])

# 2. Sample Dataset for Model Training
data = {
    'Age': np.random.randint(50, 90, 200),
    'MMSE_Score': np.random.randint(10, 30, 200),
    'MRI_Brain_Volume': np.random.uniform(500, 1500, 200),
    'Memory_Loss_Score': np.random.randint(1, 10, 200),
    'Family_History': np.random.choice([0, 1], 200),
    'Assessment_Difficulty': np.random.randint(1, 5, 200),
    'Speech_Pause_Rate': np.random.uniform(0.5, 5.0, 200),
    'Gait_Speed': np.random.uniform(0.1, 1.5, 200),
    'Sleep_Hours_Per_Day': np.random.uniform(3, 10, 200),
    'Eye_Fixation_Time': np.random.uniform(100, 800, 200),
    'HRV_Index': np.random.uniform(20, 100, 200),
    'Dementia_Level': np.random.choice(['Mild', 'Moderate', 'Severe'], 200)
}

df = pd.DataFrame(data)

# 3. Encoding categorical target variable
df['Dementia_Level'] = df['Dementia_Level'].map({'Mild': 0, 'Moderate': 1, 'Severe': 2})

# 4. Splitting dataset
X = df.drop(columns=['Dementia_Level'])
y = df['Dementia_Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Handling Class Imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 6. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 8. User Input Prediction
user_data = get_user_input()
user_data = scaler.transform(user_data)
user_prediction = best_model.predict(user_data)
prediction_labels = {0: 'Mild Dementia', 1: 'Moderate Dementia', 2: 'Severe Dementia'}

print("Predicted Dementia Level:", prediction_labels[user_prediction[0]])

# 9. Evaluation
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
