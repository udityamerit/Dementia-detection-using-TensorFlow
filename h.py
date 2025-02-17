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

# 2. Load Dataset from CSV File
# file_path = "dementia_dataset3.csv"
file_path = "dementia_dataset4.csv"
df = pd.read_csv(file_path)

# Display first few rows to see column names
print("CSV data preview:")
print(df.head())
print("\nAvailable columns:", df.columns.tolist())

# Check and identify target column (could be different from 'Dementia_Level')
# Common variations might be: 'dementia_level', 'DementiaLevel', 'Dementia Level', 'diagnosis', etc.
possible_target_columns = ['Dementia_Level', 'dementia_level', 'DementiaLevel', 'Dementia Level', 
                          'diagnosis', 'Diagnosis', 'class', 'Class', 'target', 'Target', 'label', 'Label']

target_column = None
for col in possible_target_columns:
    if col in df.columns:
        target_column = col
        print(f"Found target column: {target_column}")
        break

if target_column is None:
    raise ValueError("Could not find target column. Please check your CSV file and rename the target column or specify it manually.")

# 3. Split data into features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Check if target needs encoding (if it's not already numeric)
if y.dtype == 'object':
    print("Target values before encoding:", y.unique())
    # Perform encoding
    if len(y.unique()) == 3:  # Assuming 3 levels: Mild, Moderate, Severe
        y = y.map({val: idx for idx, val in enumerate(y.unique())})
    else:
        # Use label encoding for arbitrary number of classes
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    print("Target values after encoding:", y.unique())

# 4. Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# 5. Handling Class Imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Training set shape after SMOTE:", X_train.shape, y_train.shape)

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
print("Best hyperparameters:", grid_search.best_params_)

# 8. User Input Prediction
user_data = get_user_input()
user_data = scaler.transform(user_data)
user_prediction = best_model.predict(user_data)

# Map the prediction back to labels
if len(np.unique(y)) == 3:  # Assuming 3 levels: Mild, Moderate, Severe
    prediction_labels = {0: 'Mild Dementia', 1: 'Moderate Dementia', 2: 'Severe Dementia'}
else:
    # For other cases, just report the numeric prediction
    prediction_labels = {i: f"Class {i}" for i in range(len(np.unique(y)))}

print("Predicted Dementia Level:", prediction_labels[user_prediction[0]])

# 9. Evaluation
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and scaler
import joblib
joblib.dump(best_model, 'dementia_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully.")