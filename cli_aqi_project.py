import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')
plt.style.use('dark_background')
sns.set_palette("muted")

print("==================================================")
print("  AQI CLASSIFICATION ENGINE v2.0 (XGBOOST)  ")
print("==================================================\n")

# ---------------------------------------------------------
# 1. DATA INGESTION & MASTER MERGE
# ---------------------------------------------------------
print("[1/5] Loading and Merging Datasets to eliminate Data Drift...")
try:
    df_hist = pd.read_csv("delhi_ncr_aqi_dataset.csv")
    df_2025 = pd.read_csv("delhi-weather-aqi-2025.csv")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure dataset files are in this directory.")
    exit()

df_2025.rename(columns={
    'pm2_5': 'pm25', 'temp_c': 'temperature', 'windspeed_kph': 'wind_speed', 'aqi_index': 'aqi'
}, inplace=True)

features = ['pm25', 'pm10', 'no2', 'co', 'temperature', 'humidity', 'wind_speed']

df_hist = df_hist[features + ['aqi']].dropna()
df_2025 = df_2025[features + ['aqi']].dropna()

# Combine both datasets into one massive master dataset
df_master = pd.concat([df_hist, df_2025], ignore_index=True)

def categorize_aqi(aqi_val):
    if aqi_val <= 100: return "Good"
    elif aqi_val <= 300: return "Poor"
    else: return "Bad"

df_master['target_class'] = df_master['aqi'].apply(categorize_aqi)

# XGBoost requires numeric labels. We use LabelEncoder.
le = LabelEncoder()
df_master['target_encoded'] = le.fit_transform(df_master['target_class'])

X = df_master[features]
y = df_master['target_encoded']

# Split 80% for training, 20% for unseen testing. 
# stratify=y ensures we have a perfectly balanced ratio of Good/Poor/Bad in our test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------
print("[2/5] Generating Analytical Graphs...")
plt.figure(figsize=(10, 8))
corr_matrix = X_train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap (Pollutants & Weather)")
plt.tight_layout()
plt.savefig("1_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------
# 3. MODEL TRAINING (XGBOOST)
# ---------------------------------------------------------
print("[3/5] Training Extreme Gradient Boosting (XGBoost) Engine...")
# We add hyper-parameters to prevent overfitting and boost accuracy
model = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

joblib.dump(model, "xgboost_aqi_model.pkl")
joblib.dump(le, "label_encoder.pkl") # Save the translator too

# ---------------------------------------------------------
# 4. RIGOROUS EVALUATION
# ---------------------------------------------------------
print("[4/5] Evaluating Model Architecture...")
y_pred_encoded = model.predict(X_test)

# Translate numbers back to strings for humans to read
y_test_decoded = le.inverse_transform(y_test)
y_pred_decoded = le.inverse_transform(y_pred_encoded)

acc = accuracy_score(y_test_decoded, y_pred_decoded) * 100
print("\n--- MODEL PERFORMANCE METRICS ---")
print(f"Overall Accuracy: {acc:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test_decoded, y_pred_decoded))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=le.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"XGBoost Confusion Matrix (Accuracy: {acc:.1f}%)")
plt.ylabel('Actual Classification')
plt.xlabel('Predicted Classification')
plt.tight_layout()
plt.savefig("2_confusion_matrix.png")
plt.close()

# Feature Importance
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="viridis")
plt.title("XGBoost Feature Importance")
plt.xlabel("Relative Importance Factor")
plt.tight_layout()
plt.savefig("3_feature_importance.png")
plt.close()

# ---------------------------------------------------------
# 5. PRESENTATION CLI MODE (DYNAMIC FILTERING)
# ---------------------------------------------------------
print("\n[5/5] Launching Presentation CLI Tool...\n")

def interactive_cli(model, le, X_test, y_test_decoded):
    print("==================================================")
    print("  LIVE AQI PREDICTION SYSTEM (PRESENTATION MODE) ")
    print("==================================================")
    
    # We create a combined test dataframe to easily filter for scenarios
    test_df = X_test.copy()
    test_df['actual_aqi'] = y_test_decoded
    
    while True:
        print("\n--- Select a Test Scenario ---")
        print("1. Random Real-World Sample (Pull from Unseen Test Data)")
        print("2. Scenario: Severe Winter Smog (Dynamically pulled from data)")
        print("3. Scenario: Clear Monsoon Air (Dynamically pulled from data)")
        print("4. Manual Custom Entry")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == '5':
            print("Exiting system. Thank you.")
            break
            
        elif choice == '1':
            idx = np.random.randint(0, len(test_df))
            sample_data = test_df.iloc[[idx]][features]
            actual_class = test_df.iloc[idx]['actual_aqi']
            print("\n[!] Fetching random historical sensor reading...")
            
        elif choice == '2':
            # Dynamically find a Severe Smog day: High PM, Low Wind, Bad AQI
            smog_pool = test_df[(test_df['pm25'] > 300) & (test_df['wind_speed'] < 5) & (test_df['actual_aqi'] == 'Bad')]
            if len(smog_pool) > 0:
                # Pick a random row from this filtered pool
                sample_row = smog_pool.sample(1)
                sample_data = sample_row[features]
                actual_class = sample_row['actual_aqi'].values[0]
                print("\n[!] Dynamically extracted a 'Severe Winter Smog' event from the dataset...")
            else:
                print("\n[ERROR] Could not find a matching scenario in the test slice. Try Option 1.")
                continue
            
        elif choice == '3':
            # Dynamically find a Clear Air day: Low PM, Good AQI
            clear_pool = test_df[(test_df['pm25'] < 50) & (test_df['actual_aqi'] == 'Good')]
            if len(clear_pool) > 0:
                # Pick a random row from this filtered pool
                sample_row = clear_pool.sample(1)
                sample_data = sample_row[features]
                actual_class = sample_row['actual_aqi'].values[0]
                print("\n[!] Dynamically extracted a 'Clear Air' event from the dataset...")
            else:
                print("\n[ERROR] Could not find a matching scenario in the test slice. Try Option 1.")
                continue
            
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

        print("\n--- SENSOR INPUT DATA ---")
        for col in features:
            print(f"{col.upper().ljust(15)}: {sample_data[col].values[0]:.2f}")

        # Run Prediction (XGBoost outputs numeric, so we decode it)
        pred_encoded = model.predict(sample_data)[0]
        pred_class = le.inverse_transform([pred_encoded])[0]
        probabilities = model.predict_proba(sample_data)[0]
        
        print("\n>> ANALYZING DATA...")
        print(f">> SYSTEM PREDICTION     : ** {pred_class.upper()} **")
        if choice in ['1', '2', '3']:
             print(f">> ACTUAL RECORDED AQI   :    {actual_class.upper()}")
        
        print("\n>> Confidence Distribution:")
        for i, class_name in enumerate(le.classes_):
            print(f"   - {class_name}: {probabilities[i]*100:.1f}%")
        print("--------------------------------------------------")

# Make sure this line is at the very bottom calling the function!
interactive_cli(model, le, X_test, y_test_decoded)