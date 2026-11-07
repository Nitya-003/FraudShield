## Problem & Core Challenge

* **The Core Issue:** Financial fraud is extremely rare (usually under 0.2% of transactions), leading to massive **class imbalance**.
* **Legacy Limitation:** Traditional rule engines (e.g., *Flag if Amount > ₹50,000*) miss new fraud patterns and flag too many legitimate users.
* **ML Solution:** Supervised machine learning identifies multi-factor behavioural anomalies and outputs a continuous risk score (0 to 1) in real time.
* **Project Scope:** Academic demonstration for the Fundamentals of Machine Learning coursework.

---

## Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Feature Scaling & Preprocessing:** Scikit-Learn (`RobustScaler`, `StandardScaler`)
* **Class Imbalance Handling:** Imbalanced-learn (`SMOTE`) or model-level cost-sensitive weighting (`class_weight='balanced'`)
* **Core ML Models:** Scikit-Learn (Logistic Regression baseline, Random Forest) and LightGBM / XGBoost (champion model)
* **Model Explainability:** SHAP (SHapley Additive exPlanations)
* **Interactive UI / Demo (Optional):** Streamlit or FastAPI

---

## FraudShield MVP: Step-by-Step Architecture

**Step 1: Dataset & Ingestion**
* Use standard tabular benchmarks (e.g., Kaggle Credit Card Fraud or PaySim synthetic mobile transactions).
* Key attributes tracked: Transaction Amount, Time, Sender/Receiver IDs, Merchant Category, and Fraud Label (0 or 1).


**Step 2: Feature Engineering (Behavioural Signals)**
* **Velocity Checks:** Number of transactions made by the same user in the last 10 minutes, 1 hour, and 24 hours.
* **Amount Deviation:** Ratio of the current transaction amount relative to the user's 30-day average spend.
* **Time Anomalies:** Categorizing odd-hour nocturnal transactions (e.g., midnight to 5 AM) using cyclical sine/cosine time transforms.


**Step 3: Handling Class Imbalance**
* Never rely on standard training splits without adjustment.
* Apply algorithmic cost-weighting (`scale_pos_weight` in LightGBM or balanced weights in Random Forest) so the loss function penalizes missed frauds heavily.


**Step 4: Model Training Pipeline**
* **Baseline Model:** Logistic Regression to establish an interpretable, linear benchmark.
* **Ensemble Model:** Random Forest or LightGBM to capture non-linear interactions across velocity and amount spikes.


**Step 5: Evaluation Protocol (No Raw Accuracy)**
* **Precision-Recall AUC (PR-AUC):** Primary benchmark metric for extreme class imbalance.
* **Recall / True Positive Rate:** Measures how many actual fraud attempts were successfully caught.
* **False Positive Rate:** Ensures legitimate transactions are not aggressively rejected.


**Step 6: Real-Time Decisioning Logic**
* **Low Risk (< 0.30):** Instant transaction approval.
* **Medium Risk (0.30 – 0.75):** Step-up authentication required (e.g., OTP / 2FA trigger).
* **High Risk (> 0.75):** Immediate decline and administrative alert.


**Step 7: Explainability (SHAP Output)**
* For every flagged transaction, extract the top 3 contributing factors (e.g., *Unusual Transaction Amount*, *High 10-Minute Velocity*, *Foreign IP/Location*) for auditability and compliance.
