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

print("\nStarting Delhi NCR AQI Model Training...")

# ---------------------------------------------------------
# 1. DATA INGESTION & MASTER MERGE
# ---------------------------------------------------------
print("Loading datasets...")
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

df_master = pd.concat([df_hist, df_2025], ignore_index=True)

def categorize_aqi(aqi_val):
    if aqi_val <= 100: return "Good"
    elif aqi_val <= 300: return "Poor"
    else: return "Bad"

df_master['target_class'] = df_master['aqi'].apply(categorize_aqi)

le = LabelEncoder()
df_master['target_encoded'] = le.fit_transform(df_master['target_class'])

X = df_master[features]
y = df_master['target_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------
print("Generating evaluation graphs...")
plt.figure(figsize=(10, 8))
corr_matrix = X_train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("1_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------
# 3. MODEL TRAINING (XGBOOST)
# ---------------------------------------------------------
print("Training XGBoost classifier...")
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
joblib.dump(le, "label_encoder.pkl")

# ---------------------------------------------------------
# 4. EVALUATION
# ---------------------------------------------------------
print("Evaluating model performance on test data...\n")
y_pred_encoded = model.predict(X_test)

y_test_decoded = le.inverse_transform(y_test)
y_pred_decoded = le.inverse_transform(y_pred_encoded)

acc = accuracy_score(y_test_decoded, y_pred_decoded) * 100
print(f"Overall Accuracy: {acc:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test_decoded, y_pred_decoded))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=le.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix (Accuracy: {acc:.1f}%)")
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.tight_layout()
plt.savefig("2_confusion_matrix.png")
plt.close()

# Feature Importance
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=np.array(features)[indices], palette="viridis")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("3_feature_importance.png")
plt.close()

# ---------------------------------------------------------
# 5. PRESENTATION CLI MODE 
# ---------------------------------------------------------
def interactive_cli(model, le, X_test, y_test_decoded):
    print("\nInteractive Prediction Tool")
    print("-" * 25)
    
    test_df = X_test.copy()
    test_df['actual_aqi'] = y_test_decoded
    
    while True:
        print("\nSelect a scenario to test:")
        print("1. Random sample from test data")
        print("2. Severe winter smog event")
        print("3. Clear air event")
        print("4. Manual data entry")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == '5':
            print("Exiting tool.")
            break
            
        elif choice == '1':
            idx = np.random.randint(0, len(test_df))
            sample_data = test_df.iloc[[idx]][features]
            actual_class = test_df.iloc[idx]['actual_aqi']
            print("\nPulling random historical record...")
            
        elif choice == '2':
            smog_pool = test_df[(test_df['pm25'] > 300) & (test_df['wind_speed'] < 5) & (test_df['actual_aqi'] == 'Bad')]
            if len(smog_pool) > 0:
                sample_row = smog_pool.sample(1)
                sample_data = sample_row[features]
                actual_class = sample_row['actual_aqi'].values[0]
                print("\nPulling severe smog record from database...")
            else:
                print("\nCould not find a matching scenario in this test slice. Try Option 1.")
                continue
            
        elif choice == '3':
            clear_pool = test_df[(test_df['pm25'] < 50) & (test_df['actual_aqi'] == 'Good')]
            if len(clear_pool) > 0:
                sample_row = clear_pool.sample(1)
                sample_data = sample_row[features]
                actual_class = sample_row['actual_aqi'].values[0]
                print("\nPulling clear air record from database...")
            else:
                print("\nCould not find a matching scenario in this test slice. Try Option 1.")
                continue
            
        elif choice == '4':
            try:
                print("\nEnter sensor readings:")
                pm25 = float(input("PM2.5: "))
                pm10 = float(input("PM10: "))
                no2 = float(input("NO2: "))
                co = float(input("CO: "))
                temp = float(input("Temperature (°C): "))
                humidity = float(input("Humidity (%): "))
                wind = float(input("Wind Speed (kph): "))
                sample_data = pd.DataFrame([[pm25, pm10, no2, co, temp, humidity, wind]], columns=features)
                actual_class = "N/A (Manual Entry)"
            except ValueError:
                print("\nInvalid input. Please enter numbers only.")
                continue
        else:
            print("Invalid choice.")
            continue

        print("\nInput values:")
        for col in features:
            print(f"  {col}: {sample_data[col].values[0]:.2f}")

        pred_encoded = model.predict(sample_data)[0]
        pred_class = le.inverse_transform([pred_encoded])[0]
        probabilities = model.predict_proba(sample_data)[0]
        
        print(f"\nPredicted Category: {pred_class.upper()}")
        if choice in ['1', '2', '3']:
             print(f"Actual Category:    {actual_class.upper()}")
        
        print("\nModel Confidence:")
        for i, class_name in enumerate(le.classes_):
            print(f"  {class_name}: {probabilities[i]*100:.1f}%")
        print("-" * 25)

interactive_cli(model, le, X_test, y_test_decoded)