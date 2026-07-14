"""
Credit Card Approval Prediction — Flask Application
Loads the tuned V3 XGBoost model and serves a prediction form.
"""
import json
import joblib
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request

app = Flask(__name__)

# ----------------------------------------------------------------------
# Load model artifacts once at startup
# ----------------------------------------------------------------------
# NOTE: loaded via XGBoost's native format (.json), not joblib/pickle.
# The V3 model was originally saved with joblib.dump (a pickle of the
# XGBClassifier), but pickled boosters can fail to deserialize across
# machines/XGBoost versions ("input stream corrupted"). It's been
# re-exported here via model.save_model() to the native .json format,
# which is version-portable, same as V2.
MODEL = xgb.XGBClassifier()
MODEL.load_model('model/best_model_v3_tuned.json')
FEATURE_COLUMNS = joblib.load('model/feature_columns_v3.pkl')
with open('model/model_metadata_v3.json') as f:
    METADATA = json.load(f)
THRESHOLD = METADATA['threshold']  # 0.29 — tuned via cross-validation, NOT the default 0.5

EDU_ORDER = {
    'Lower secondary': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Higher education': 3,
    'Academic degree': 4,
}
INCOME_TYPES = ['Commercial associate', 'Pensioner', 'State servant', 'Student', 'Working']
FAMILY_STATUSES = ['Civil marriage', 'Married', 'Separated', 'Single / not married', 'Widow']
OCCUPATIONS = ['Accountants', 'Cleaning staff', 'Cooking staff', 'Core staff', 'Drivers',
               'HR staff', 'High skill tech staff', 'IT staff', 'Laborers', 'Low-skill Laborers',
               'Managers', 'Medicine staff', 'Private service staff', 'Realty agents',
               'Sales staff', 'Secretaries', 'Security staff', 'Waiters/barmen staff', 'Unknown']
HOUSING_TYPES = ['House / apartment', 'With parents', 'Municipal apartment',
                  'Rented apartment', 'Office apartment', 'Co-op apartment']

INCOME_MIN = 27000.0
INCOME_MAX = 562500.0


def validate_form(form):
    try:
        income = float(form.get('income', '0'))
    except ValueError:
        return 'Annual income must be a valid number.'

    # if income < INCOME_MIN or income > INCOME_MAX:
    #     return f'Annual income must be between {int(INCOME_MIN):,} and {int(INCOME_MAX):,}.'

    try:
        age = int(form.get('age', '0'))
    except ValueError:
        return 'Age must be a whole number.'
    if age < 18 or age > 100:
        return 'Age must be between 18 and 100.'

    try:
        family_size = float(form.get('family_size', '0'))
    except ValueError:
        return 'Family size must be a valid number.'
    if family_size < 1:
        return 'Family size must be at least 1.'

    return None


def build_feature_vector(form):
    row = {col: 0 for col in FEATURE_COLUMNS}

    row['CODE_GENDER'] = 1 if form['gender'] == 'M' else 0
    row['FLAG_OWN_CAR'] = 1 if form['own_car'] == 'Y' else 0
    row['FLAG_OWN_REALTY'] = 1 if form['own_realty'] == 'Y' else 0
    row['AMT_INCOME_TOTAL'] = float(form['income'])
    row['NAME_EDUCATION_TYPE'] = EDU_ORDER[form['education']]
    row['FLAG_WORK_PHONE'] = 1 if form['work_phone'] == 'Y' else 0
    row['FLAG_PHONE'] = 1 if form['phone'] == 'Y' else 0
    row['FLAG_EMAIL'] = 1 if form['email'] == 'Y' else 0
    row['AGE_YEARS'] = int(form['age'])

    not_employed = form['currently_employed'] == 'N'
    row['NOT_EMPLOYED_FLAG'] = 1 if not_employed else 0
    row['EMPLOYED_YEARS'] = 0.0 if not_employed else float(form['years_employed'])

    children = float(form['children'])
    family_size = max(float(form['family_size']), 1)
    row['DEPENDENTS_RATIO'] = children / family_size
    row['FAMILY_SIZE'] = family_size

    for opt in INCOME_TYPES:
        col = f'NAME_INCOME_TYPE_{opt}'
        if col in row:
            row[col] = 1 if form['income_type'] == opt else 0
    for opt in FAMILY_STATUSES:
        col = f'NAME_FAMILY_STATUS_{opt}'
        if col in row:
            row[col] = 1 if form['family_status'] == opt else 0
    for opt in OCCUPATIONS:
        col = f'OCCUPATION_TYPE_{opt}'
        if col in row:
            row[col] = 1 if form['occupation'] == opt else 0
    for opt in HOUSING_TYPES:
        col = f'NAME_HOUSING_TYPE_{opt}'
        if col in row:
            row[col] = 1 if form['housing_type'] == opt else 0

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
@app.route('/')
def open_page():
    return render_template('open.html')

@app.route('/prediction', methods=['GET'])
def index():
    return render_template(
        'index.html',
        income_types=INCOME_TYPES,
        family_statuses=FAMILY_STATUSES,
        occupations=OCCUPATIONS,
        housing_types=HOUSING_TYPES,
        education_levels=list(EDU_ORDER.keys()),
    )


@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    error_message = validate_form(form)
    if error_message:
        return render_template(
            'result.html',
            decision='Reject',
            proba_approve=0.0,
            proba_reject=100.0,
            threshold=round(THRESHOLD * 100, 1),
            applicant_name=form.get('applicant_name', 'Applicant'),
            error_message=error_message,
        )

    try:
        income = float(form.get('income', '0'))
    except ValueError:
        income = 0.0

    if income < INCOME_MIN:
        return render_template(
            'result.html',
            decision='Reject',
            proba_approve=0.0,
            proba_reject=100.0,
            threshold=round(THRESHOLD * 100, 1),
            applicant_name=form.get('applicant_name', 'Applicant'),
        )

    vector = build_feature_vector(form)
    proba_approve = float(MODEL.predict_proba(vector)[:, 1][0])
    proba_reject = 1 - proba_approve
    decision = 'Approve' if proba_approve >= THRESHOLD else 'Reject'

    return render_template(
        'result.html',
        decision=decision,
        proba_approve=round(proba_approve * 100, 1),
        proba_reject=round(proba_reject * 100, 1),
        threshold=round(THRESHOLD * 100, 1),
        applicant_name=form.get('applicant_name', 'Applicant'),
    )


if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
