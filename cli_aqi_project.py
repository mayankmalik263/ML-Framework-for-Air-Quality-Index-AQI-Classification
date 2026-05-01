import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Set aesthetic style for graphs
plt.style.use('dark_background') # Looks highly professional for tech presentations
sns.set_palette("muted")

print("==================================================")
print("  AQI CLASSIFICATION ENGINE & CLI TOOL  ")
print("==================================================\n")

# ---------------------------------------------------------
# 1. DATA INGESTION & ALIGNMENT
# ---------------------------------------------------------
print("[1/5] Loading and Aligning Datasets...")
try:
    df_hist = pd.read_csv("delhi_ncr_aqi_dataset.csv")
    df_2025 = pd.read_csv("delhi-weather-aqi-2025.csv")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure dataset files are in this directory.")
    exit()

# Standardize 2025 columns
df_2025.rename(columns={
    'pm2_5': 'pm25', 'temp_c': 'temperature', 'windspeed_kph': 'wind_speed', 'aqi_index': 'aqi'
}, inplace=True)

features = ['pm25', 'pm10', 'no2', 'co', 'temperature', 'humidity', 'wind_speed']

df_hist = df_hist[features + ['aqi']].dropna()
df_2025 = df_2025[features + ['aqi']].dropna()

def categorize_aqi(aqi_val):
    if aqi_val <= 100: return "Good"
    elif aqi_val <= 300: return "Poor"
    else: return "Bad"

df_hist['target_class'] = df_hist['aqi'].apply(categorize_aqi)
df_2025['target_class'] = df_2025['aqi'].apply(categorize_aqi)

X_train, y_train = df_hist[features], df_hist['target_class']
X_test, y_test = df_2025[features], df_2025['target_class']

# ---------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (VISUALIZATIONS)
# ---------------------------------------------------------
print("[2/5] Generating Analytical Graphs (Saving to disk)...")

# Graph 1: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = X_train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap (Pollutants & Weather)")
plt.tight_layout()
plt.savefig("1_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------
# 3. MODEL TRAINING
# ---------------------------------------------------------
print("[3/5] Training Random Forest Classifier Engine...")
# We use class_weight='balanced' to ensure the model doesn't favor the majority class
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "cli_aqi_model.pkl")

# ---------------------------------------------------------
# 4. RIGOROUS EVALUATION & METRICS
# ---------------------------------------------------------
print("[4/5] Evaluating on Unseen 2025 Temporal Data...")
y_pred = model.predict(X_test)

print("\n--- MODEL PERFORMANCE METRICS ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# Graph 2: Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=["Good", "Poor", "Bad"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Good", "Poor", "Bad"], yticklabels=["Good", "Poor", "Bad"])
plt.title("Confusion Matrix: Predicted vs Actual Classes")
plt.ylabel('Actual Classification')
plt.xlabel('Predicted Classification')
plt.tight_layout()
plt.savefig("2_confusion_matrix.png")
plt.close()

# Graph 3: Feature Importance
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="viridis")
plt.title("Feature Importance: What Drives the AQI Classification?")
plt.xlabel("Relative Importance Factor")
plt.tight_layout()
plt.savefig("3_feature_importance.png")
plt.close()

print("\n[!] Check your folder: 3 graphs have been generated for your presentation slides.")

# ---------------------------------------------------------
# 5. PRESENTATION CLI MODE
# ---------------------------------------------------------
print("\n[5/5] Launching Presentation CLI Tool...\n")

def interactive_cli(model, X_test, y_test):
    print("==================================================")
    print("  LIVE AQI PREDICTION SYSTEM (PRESENTATION MODE) ")
    print("==================================================")
    
    while True:
        print("\n--- Select a Test Scenario ---")
        print("1. Random Real-World Sample (Pull from 2025 Data)")
        print("2. Scenario: Severe Winter Smog (High PM, Low Wind)")
        print("3. Scenario: Post-Monsoon Morning (Clear Air)")
        print("4. Manual Custom Entry")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == '5':
            print("Exiting system. Thank you.")
            break
            
        elif choice == '1':
            # Pick a random row from the unseen test data
            idx = np.random.randint(0, len(X_test))
            sample_data = X_test.iloc[[idx]]
            actual_class = y_test.iloc[idx]
            print("\n[!] Fetching random historical sensor reading...")
            
        elif choice == '2':
            # Realistic "Bad" Day in Delhi
            sample_data = pd.DataFrame([[350.5, 480.0, 85.2, 2.5, 12.0, 85.0, 2.1]], columns=features)
            actual_class = "Bad"
            print("\n[!] Loading 'Severe Winter Smog' parameters...")
            
        elif choice == '3':
            # Realistic "Good" Day in Delhi
            sample_data = pd.DataFrame([[45.0, 80.0, 25.0, 0.8, 28.0, 60.0, 15.5]], columns=features)
            actual_class = "Good"
            print("\n[!] Loading 'Post-Monsoon Morning' parameters...")
            
        elif choice == '4':
            try:
                print("\n(Tip: PM2.5 in Delhi typically ranges from 30 to 400+)")
                pm25 = float(input("PM2.5 Level (μg/m3)    : "))
                pm10 = float(input("PM10 Level (μg/m3)     : "))
                no2 = float(input("NO2 Level (μg/m3)      : "))
                co = float(input("CO Level (μg/m3)       : "))
                temp = float(input("Temperature (°C)       : "))
                humidity = float(input("Humidity (%)           : "))
                wind = float(input("Wind Speed (kph)       : "))
                
                sample_data = pd.DataFrame([[pm25, pm10, no2, co, temp, humidity, wind]], columns=features)
                actual_class = "Unknown (Manual Entry)"
            except ValueError:
                print("\n[ERROR] Invalid input. Please enter numerical values only.")
                continue
        else:
            print("Invalid choice.")
            continue

        # Display the Inputs to the audience
        print("\n--- SENSOR INPUT DATA ---")
        for col in features:
            print(f"{col.upper().ljust(15)}: {sample_data[col].values[0]:.2f}")

        # Run Prediction
        pred_class = model.predict(sample_data)[0]
        probabilities = model.predict_proba(sample_data)[0]
        
        print("\n>> ANALYZING DATA...")
        print(f">> SYSTEM PREDICTION     : ** {pred_class.upper()} **")
        if choice in ['1', '2', '3']:
             print(f">> ACTUAL RECORDED AQI   :    {actual_class.upper()}")
        
        # Show confidence distribution
        print("\n>> Confidence Distribution:")
        for i, c in enumerate(model.classes_):
            print(f"   - {c}: {probabilities[i]*100:.1f}%")
        print("--------------------------------------------------")

# Start the interactive loop (Make sure to pass the required variables!)
interactive_cli(model, X_test, y_test)