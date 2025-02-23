import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

def save_plot(fig, filename):
    """Save plot to the results directory"""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(os.path.join(results_dir, filename))

def plot_training_testing_metrics(model, X_train, X_test, y_train, y_test):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score', color='blue', marker='o')
    ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation Score', color='green', marker='s')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy Score')
    ax.set_title('Learning Curves')
    ax.legend(loc='lower right')
    ax.grid(True)
    save_plot(fig, 'learning_curve.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    save_plot(fig, 'confusion_matrix.png')
    plt.show()

# Visualization Functions
def plot_training_testing_metrics(model, X_train, X_test, y_train, y_test):
    """Plot training and testing accuracy/error metrics"""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5,
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    
    plt.plot(train_sizes, test_mean, label='Cross-validation Score', color='green', marker='s')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix with labels"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_feature_importance(model, feature_names):
    """Plot feature importance from the Random Forest model"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_hyperparameter_tuning(grid_search, param_name):
    """Plot the effect of a hyperparameter on model performance"""
    results = pd.DataFrame(grid_search.cv_results_)
    param_col = f'param_{param_name}'
    scores = results.groupby(param_col)['mean_test_score'].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(scores.index, scores.values, marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Mean CV Score')
    plt.title(f'Effect of {param_name} on Model Performance')
    plt.grid(True)
    plt.show()

# Main Program
def get_user_input():
    user_data_standard = {}
    
    user_data_standard['Age'] = int(input("Enter Age: "))
    user_data_standard['MMSE_Score'] = int(input("Enter MMSE Score (0-30): "))
    user_data_standard['MRI_Brain_Volume'] = float(input("Enter MRI Brain Volume (cubic mm): "))
    user_data_standard['Memory_Loss_Score'] = int(input("Enter Memory Loss Score (1-10): "))
    user_data_standard['Family_History'] = int(input("Family History of Dementia? (0 = No, 1 = Yes): "))
    user_data_standard['Assessment_Difficulty'] = int(input("Enter Assessment Difficulty (1-5): "))
    user_data_standard['Speech_Pause_Rate'] = float(input("Enter Speech Pause Rate (words per minute): "))
    user_data_standard['Gait_Speed'] = float(input("Enter Gait Speed (meters per second): "))
    user_data_standard['Sleep_Hours_Per_Day'] = float(input("Enter Average Sleep Hours Per Day: "))
    user_data_standard['Eye_Fixation_Time'] = float(input("Enter Eye Fixation Time (milliseconds): "))
    user_data_standard['HRV_Index'] = float(input("Enter Heart Rate Variability Index (ms): "))
    
    # Convert to DataFrame with column names matching the CSV
    user_data_csv = {}
    for std_name, value in user_data_standard.items():
        csv_name = column_mapping[std_name]
        user_data_csv[csv_name] = value
    
    return pd.DataFrame([user_data_csv])

def main():
    # Load Dataset
    file_path = "dementia_dataset4.csv"
    df = pd.read_csv(file_path)
    
    print("CSV data preview:")
    print(df.head())
    print("\nAvailable columns:", df.columns.tolist())
    
    # Identify target column
    possible_target_columns = ['Dementia_Level', 'dementia_level', 'DementiaLevel', 'Dementia Level', 
                              'diagnosis', 'Diagnosis', 'class', 'Class', 'target', 'Target', 'label', 'Label']
    
    target_column = None
    for col in possible_target_columns:
        if col in df.columns:
            target_column = col
            print(f"Found target column: {target_column}")
            break
    
    if target_column is None:
        raise ValueError("Could not find target column. Please check your CSV file.")
    
    # Store original feature columns
    original_feature_columns = [col for col in df.columns if col != target_column]
    print("Feature columns from CSV:", original_feature_columns)
    
    # Create column mapping
    global column_mapping
    column_mapping = {
        'Age': next((col for col in original_feature_columns if 'age' in col.lower()), 'Age'),
        'MMSE_Score': next((col for col in original_feature_columns if 'mmse' in col.lower() or ('score' in col.lower() and 'memory' not in col.lower())), 'MMSE_Score'),
        'MRI_Brain_Volume': next((col for col in original_feature_columns if 'brain' in col.lower() and 'volume' in col.lower()), 'MRI_Brain_Volume'),
        'Memory_Loss_Score': next((col for col in original_feature_columns if 'memory' in col.lower() and 'loss' in col.lower()), 'Memory_Loss_Score'),
        'Family_History': next((col for col in original_feature_columns if 'family' in col.lower() and 'history' in col.lower()), 'Family History of Dementia'),
        'Assessment_Difficulty': next((col for col in original_feature_columns if 'assessment' in col.lower() and 'difficulty' in col.lower()), 'Assessment Difficulty'),
        'Speech_Pause_Rate': next((col for col in original_feature_columns if 'speech' in col.lower() and 'pause' in col.lower()), 'Speech_Pause_Rate'),
        'Gait_Speed': next((col for col in original_feature_columns if 'gait' in col.lower() and 'speed' in col.lower()), 'Gait Speed'),
        'Sleep_Hours_Per_Day': next((col for col in original_feature_columns if 'sleep' in col.lower() and 'hour' in col.lower()), 'Average Sleep Hours Per Day'),
        'Eye_Fixation_Time': next((col for col in original_feature_columns if 'eye' in col.lower() and 'fixation' in col.lower()), 'Eye Fixation Time'),
        'HRV_Index': next((col for col in original_feature_columns if 'hrv' in col.lower()), 'HRV_Index')
    }
    
    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target if needed
    if y.dtype == 'object':
        print("Target values before encoding:", y.unique())
        unique_target_values = sorted(y.unique())
        target_mapping = {val: idx for idx, val in enumerate(unique_target_values)}
        y = y.map(target_mapping)
        print("Target mapping:", target_mapping)
        prediction_map = {idx: label for label, idx in target_mapping.items()}
    else:
        prediction_map = {i: f"Class {i}" for i in range(len(np.unique(y)))}
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("Training set shape after SMOTE:", X_train.shape, y_train.shape)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)
    
    train_scores = []
    test_scores = []
    values = [i for i in range(1,1000,30)]
    
    for i in values:
        model = RandomForestClassifier(n_estimators=i, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        train_yhat = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_yhat)
        
        test_yhat = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_yhat)
        
        train_scores.append(train_acc)
        test_scores.append(test_acc)
        
        print('>>%d, train: %.3f, test: %.3f' % (i, train_acc*100, test_acc*100))

    # Generate visualizations
    print("\nGenerating model performance visualizations...")
    plot_training_testing_metrics(best_model, X_train_scaled, X_test_scaled, y_train, y_test)
    
    y_pred = best_model.predict(X_test_scaled)
    plot_confusion_matrix(y_test, y_pred, classes=list(prediction_map.values()))
    
    plot_feature_importance(best_model, X.columns)
    
    for param in param_grid.keys():
        plot_hyperparameter_tuning(grid_search, param)
    
    # Get user input and make prediction
    print("\nEnter patient information:")
    user_data = get_user_input()
    
    # Ensure user_data has same columns as training data
    missing_cols = set(X.columns) - set(user_data.columns)
    if missing_cols:
        print(f"Warning: Missing columns in user data: {missing_cols}")
        for col in missing_cols:
            user_data[col] = 0
    
    
    
    user_data = user_data[X.columns]
    user_data_scaled = scaler.transform(user_data)
    user_prediction = best_model.predict(user_data_scaled)
    
    print("\nPredicted Dementia Level:", prediction_map[user_prediction[0]])
    
    # Model evaluation
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save model and related objects
    joblib.dump(best_model, 'dementia_prediction_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump({'original_columns': X.columns.tolist(), 'column_mapping': column_mapping}, 'column_info.pkl')
    print("\nModel, scaler, and column information saved successfully.")

if __name__ == "__main__":
    main()