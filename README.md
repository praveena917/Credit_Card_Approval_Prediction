# 💳 Credit Card Approval Prediction

> A machine-learning approach to credit card approval decisioning, replacing slow, inconsistent manual and rule-based review with a data-driven classification model. The project defines a full pipeline — from raw applicant and credit-bureau data through a vintage-corrected target label, feature engineering, cross-validated model tuning, and deployment — culminating in a Flask web application that returns a real-time **Approve / Reject** decision with a disclosed confidence level.

> **📌 Repository Status**
>
> This repository currently contains the project's **planning and documentation deliverables** (PDF reports for each SDLC phase, listed below). The application design described in this README — Flask app, model training scripts, HTML templates — is fully specified in those reports, but the source code itself is not yet committed here.

---

# 📑 Table of Contents

- Problem Statement
- Proposed Solution
- Key Features
- Tech Stack
- Dataset
- Solution Architecture
- Planned Application Structure
- Intended Run Instructions
- Known Limitations & Roadmap
- Non-Functional Targets
- Repository Contents
- License

---

# 📌 Problem Statement

Manual, rule-based credit approval review is slow and inconsistent across reviewers. It gets harder still at scale — banks and financial institutions process thousands of applications a day, each with numerous financial risk factors, making manual evaluation time-consuming and error-prone.

A secondary, more subtle problem motivates the modeling approach: naively labeling an applicant "risky" based on *ever* having a serious delinquency — regardless of how long the account has been observed — mislabels short-history applicants as safe simply because they haven't been tracked long enough. This bakes label noise into the very data used to train and evaluate approval models.

---

# 💡 Proposed Solution

- **Vintage (cohort) analysis** is used to build a cleaner target label: only applicants with **≥24 months of observed history** are labeled, removing the bias described above. This alone improved cross-validated test F1 for the reject class from **0.345 to 0.410 (~+19% relative)**, with precision and recall improving together.

- **Baseline comparison** across four classifiers — Logistic Regression, Decision Tree, Random Forest, and XGBoost — with default hyperparameters, used as an informational benchmark.

- **Cross-validated hyperparameter tuning** (5-fold × 40-candidate `RandomizedSearchCV`) of XGBoost, optimized for F1 on the minority "reject" class.

- **Threshold selection from out-of-fold probabilities** rather than the default **0.5**, validated once on an untouched hold-out test set. The tuned operating threshold is **0.29**.

- **Real-time serving** of the final model through a Flask web form for single-applicant predictions, with the model's actual precision/recall disclosed on the result page so predictions aren't presented as more reliable than they are.

---

# ✨ Key Features

| Feature | Description |
|:---------|:------------|
| Applicant data input form | Web form capturing 18 applicant attributes — demographics, income, employment, housing, contact details |
| Dynamic employment fields | Form fields toggle (e.g. "years employed") based on employment status |
| Feature vector construction | Converts raw form input into the exact 46-column encoded vector the model expects (one-hot/ordinal encoding, engineered ratios) |
| ML-based approval prediction | Loads the tuned XGBoost classifier and scores the applicant's approval probability |
| Tuned decision threshold | Applies the cross-validated threshold (**0.29**) instead of the default **0.5**, chosen to better catch risky applicants given the ~1.7% reject rate |
| Result visualization | SVG arc gauge showing **P(Approve)** / **P(Reject)** and the decision threshold used |
| Transparency disclosure | Plain-language model performance (precision/recall) shown alongside the prediction |

---

# 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| **Frontend** | HTML, CSS, Bootstrap, JavaScript | Responsive form for entering applicant details and viewing predictions |
| **Backend** | Python, Flask | Handles requests, builds the feature vector, loads the trained model, returns predictions |
| **ML / Data** | pandas, numpy, scikit-learn, XGBoost, joblib | Preprocessing, model training/tuning, inference |
| **Data Storage** | CSV files | Applicant records and credit-bureau records; no database engine is used |
| **Development Environment** | Jupyter Notebook (EDA), PyCharm / VS Code | Analysis and development |

---

# 📊 Dataset

The project uses the Kaggle **"Credit Card Approval Prediction"** dataset.

### Dataset Files

| File | Description |
|:-----|:------------|
| `application_record.csv` | 438,557 applicant records |
| `credit_record.csv` | 1,048,575 monthly credit-bureau records |

After linking the two files:

- **36,457** applicants have bureau history.
- **13,153** applicants meet the **≥24-month vintage threshold**.
- These records form the final modeling dataset with **46 engineered and encoded features**.

---

# 🏗️ Solution Architecture

The application is designed as a lightweight, self-contained Flask application rather than a distributed microservice architecture.

```
                     Credit Card Approval Prediction System

┌──────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                           │
│      HTML Form (index.html) & Result Page (result.html)          │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Flask Routing Layer                         │
│                   Flask / Werkzeug Application                   │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Core Logic Service                           │
│  • Feature Engineering                                           │
│  • 46-column Feature Vector Construction                         │
│  • XGBoost Inference                                             │
│  • Threshold = 0.29                                              │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Data / Storage Layer                         │
│  CSV Dataset                                                     │
│  Trained XGBoost Model                                           │
│  Feature Columns                                                 │
│  Metadata                                                        │
└──────────────────────────────────────────────────────────────────┘
```

### Not Currently in Scope

- Authentication / Authorization
- External API integrations (e.g. payment or bureau APIs)
- Production WSGI / Reverse Proxy Layer (Gunicorn + Nginx recommended for production)

---

# 📂 Planned Application Structure

The following structure is described in the project documentation. The source code has **not yet been committed** to this repository.

```text
credit_card_approval_project/
│
├── app.py
├── requirements.txt
├── epic1_ER_diagram.jpeg
├── epic3_preprocessing_v3_vintage.py
├── epic4_model_development_v3.py
├── Epic2(Visualisation & Analysis).ipynb
│
├── templates/
│   ├── index.html
│   ├── open.html
│   └── result.html
│
├── static/
│   └── style.css
│
├── model/
│   ├── best_model_v3_tuned.json
│   ├── feature_columns_v3.pkl
│   └── model_metadata_v3.json
│
└── data/
    ├── cleaned_credit_data_v3_vintage.csv
    └── credit_card_approval_dataset/
```

---

# 🚀 Intended Run Instructions

### Prerequisites

- Python **3.9+**
- pip

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/praveena917/Credit_Card_Approval_Prediction.git

cd Credit_Card_Approval_Prediction
```

### 2️⃣ Create a Virtual Environment *(Optional)*

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Start the Application

```bash
python app.py
```

### 5️⃣ Open the Application

Visit

```
http://localhost:5000
```

Fill in the applicant details and click **Predict**.

---

# ⚠️ Known Limitations & Roadmap

## Current Limitations

| Limitation | Notes |
|:-----------|:------|
| Modest precision/recall on the reject class (~42% / ~40%) | Driven by severe class imbalance (~1.7% rejections); the result page discloses this so it's used as one input, not a final verdict |
| Static model, no retraining pipeline | Model is trained on a CSV snapshot and loaded once at startup; predictions can drift over time |
| No authentication, logging, or rate limiting | Not yet suitable for direct multi-user production exposure without hardening |
| No automated tests or error handling | Bad input or a missing model file would currently raise an unhandled exception |

### Planned Scalability Upgrades

- Deploy behind **Gunicorn/uWSGI + Nginx**
- Containerize the application using **Docker**
- Migrate flat CSV storage to a managed relational database
- Add caching / asynchronous serving (e.g. FastAPI + Uvicorn, or IBM Watson Machine Learning) for higher concurrency
- Add HTTPS, request authentication, and comprehensive input validation

---

# 🎯 Non-Functional Targets

| Target | Goal |
|:-------|:-----|
| ⚡ Prediction Response Time | Within **2 seconds** of submission |
| 👥 Concurrent Users | Support **100+** simultaneous prediction requests |
| 🔒 Security | Secure transmission (HTTPS) and restricted access to sensitive applicant data |
| 📈 Availability | **99% uptime** during deployment |
| ✅ Usability | A new user can complete a prediction in under **2 minutes** |

---

# 📁 Repository Contents

This repository currently contains the project's **phase-wise planning and documentation deliverables**.

| Phase | Contents |
|:------|:---------|
| **1. Brainstorming & Ideation** | Problem statement, empathy map, idea prioritization |
| **2. Requirement Analysis** | Customer journey map, data flow diagram, solution requirements, technology stack |
| **3. Project Design Phase** | Problem–solution fit, proposed solution, solution architecture |
| **4. Project Planning Phase** | Project planning |
| **5. Project Development Phase** | Coding & solution summary, code quality checklist, functional feature list |
| **6. Project Testing** | Performance testing |
| **7. Project Documentation** | Executable file checklist & structure, sample project documentation |
| **8. Project Demonstration** | Demo planning, feature demonstration, scalability & future plan, team involvement, communication 

