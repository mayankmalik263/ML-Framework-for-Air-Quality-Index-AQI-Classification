# 🌍 Delhi NCR Air Quality Index (AQI) Classification Engine

## 📊 Executive Summary
This project is an Applied Machine Learning framework designed to classify the Air Quality Index (AQI) of the Delhi NCR region into three distinct health-risk tiers: **Good (0-100)**, **Poor (101-300)**, and **Bad (>300)**. 

Rather than relying on post-regression thresholding, this engine utilizes a direct multi-class classification architecture, taking localized atmospheric pollutants (PM2.5, PM10, NO2, CO) and meteorological conditions (Temperature, Humidity, Wind Speed) as inputs to map complex, non-linear environmental interactions.

---

## 🚀 Project Evolution & Architectural Iterations
Throughout the development lifecycle, several major architectural upgrades were implemented to transition the project from a theoretical baseline to a presentation-ready, robust system:

1. **Algorithm & Data Pipeline Overhaul (Addressing Data Drift):**
   * *Initial State:* A Random Forest classifier trained on pre-2025 data and tested on 2025 data, which suffered from temporal data drift (yielding ~57% accuracy).
   * *Upgrade:* Combined historical and recent datasets into a unified "Master Dataset" and transitioned the core engine to **XGBoost (Extreme Gradient Boosting)**. By utilizing an 80/20 stratified split on the unified data, the model successfully learned cross-temporal patterns, dramatically raising predictive accuracy.
2. **Dynamic Real-World Sampling (CLI Upgrade):**
   * *Initial State:* The Presentation CLI used hardcoded, theoretical float values for environmental scenarios.
   * *Upgrade:* Implemented a dynamic query system. When testing "Severe Winter Smog" or "Clear Air" scenarios, the CLI now queries the 20% *unseen* test dataset to extract a genuine historical event matching extreme weather parameters. This proves the model's validity against real, messy data.
3. **Refactored Output Formatting:**
   * *Initial State:* Terminal output featured sensationalized formatting.
   * *Upgrade:* Completely refactored the CLI UI to adopt a clean, minimalist, and highly professional aesthetic suitable for academic defense and presentations.

---

## 🛠 Features
* **XGBoost Inference Engine:** Handles non-linear feature interactions (e.g., wind speed vs. pollutant density) with high precision.
* **Automated Data Harmonization:** Automatically cleans, maps, and aligns legacy historical datasets with 2025 meteorological data.
* **Automated Visualization Pipeline:** Generates presentation-ready PNGs upon execution:
  * `1_correlation_heatmap.png`
  * `2_confusion_matrix.png`
  * `3_feature_importance.png`
* **Interactive Presentation CLI:** A built-in command-line interface designed specifically for live academic demonstrations, featuring live inference and confidence distribution percentages.

---

## 💻 Tech Stack
* **Language:** Python 3.9+
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `xgboost`, `scikit-learn`
* **Serialization:** `joblib` (for O(1) instantaneous model loading)
* **Visualization:** `matplotlib`, `seaborn`

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   
```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
   ```

3. **Ensure Datasets are Present:**
   Verify that `delhi_ncr_aqi_dataset.csv` and `delhi-weather-aqi-2025.csv` are located in the root directory.

---

## 🖥 Usage: Running the CLI Engine

To train the model, generate evaluation graphs, and launch the interactive prediction tool, run:
```bash
python cli_aqi_project.py
```

### The Presentation Flow:
Upon running the script, the engine will process the data, train the XGBoost model, and print a rigorous **Classification Report** (Accuracy, Precision, Recall, F1-Score) to the terminal. 

It will then launch the Interactive CLI tool:
```text
Interactive Prediction Tool
-------------------------

Select a scenario to test:
1. Random sample from test data
2. Severe winter smog event
3. Clear air event
4. Manual data entry
5. Exit
```
*Tip for Presenters: Select Options `1`, `2`, or `3` to instantly pull unseen, real-world data rows to prove the model's accuracy live.*

---

## 📊 Evaluation & Metrics
The model prioritizes the **Macro F1-Score** and **Precision** on the "Bad" category. In air quality modeling, false negatives (predicting "Good" when the air is "Bad") carry a high public health cost. The model is specifically evaluated on its ability to define the complex boundary between "Poor" and "Bad" index classifications based on atmospheric stagnation.