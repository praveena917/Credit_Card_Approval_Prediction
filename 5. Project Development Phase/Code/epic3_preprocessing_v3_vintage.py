"""
Epic 3 — Data Cleaning & Preprocessing (v3, vintage-analysis label)
Credit Card Approval Prediction

WHAT'S DIFFERENT FROM V2:
  V1/V2 labeled an applicant "risky" if they were EVER 60+ days overdue,
  regardless of how long their credit history actually was. Vintage
  analysis on credit_record.csv showed this is a real problem: cumulative
  bad-rate curves, grouped by account age, never flatten within the
  observed window (they're still climbing at 60 months on book). This
  means applicants with SHORT credit histories are being labeled "good"
  partly just because they haven't been observed long enough for risk to
  reveal itself yet -- not because they're actually low-risk. That's
  label noise baked into the majority class.

  FIX: only label applicants who have at least MOB_THRESHOLD (24) months
  of observed credit history. This shrinks the dataset (13,153 vs 36,457
  rows) but produces a cleaner, more trustworthy label.

VALIDATED RESULT (cross-validated, held-out test set, XGBoost):
  V2 (unfiltered label):  F1=0.345  (precision=0.357, recall=0.333)
  V3 (vintage-filtered):  F1=0.410  (precision=0.418, recall=0.402)
  -> +19% relative improvement, with BOTH precision and recall improving
     together -- a strong sign the label itself is genuinely cleaner,
     not just a different precision/recall tradeoff point.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

MOB_THRESHOLD = 24  # minimum months-on-book required to trust the label

# ----------------------------------------------------------------------
# Setup & Load Data
# ----------------------------------------------------------------------
app_df = pd.read_csv(r"C:\Users\krish\Downloads\credit_card_approval_dataset\application_record.csv")
credit_df = pd.read_csv(r"C:\Users\krish\Downloads\credit_card_approval_dataset\credit_record.csv")

app_df = app_df.drop_duplicates(subset='ID', keep='first')

# ----------------------------------------------------------------------
# Vintage analysis setup: origination + months-on-book (MOB)
# ----------------------------------------------------------------------
# origination = the applicant's oldest record (most negative MONTHS_BALANCE)
origination = credit_df.groupby('ID')['MONTHS_BALANCE'].min().rename('origination')
credit_df = credit_df.merge(origination, on='ID')

# MOB = 0 at account origination, increasing toward the present
credit_df['MOB'] = credit_df['MONTHS_BALANCE'] - credit_df['origination']
credit_df['is_risky'] = credit_df['STATUS'].isin(['2', '3', '4', '5']).astype(int)

max_mob = credit_df.groupby('ID')['MOB'].max().rename('max_mob')

# ----------------------------------------------------------------------
# Build the Target Label -- ONLY from applicants with enough history
# ----------------------------------------------------------------------
eligible_ids = max_mob[max_mob >= MOB_THRESHOLD].index
print(f"Applicants with >= {MOB_THRESHOLD} months on book: "
      f"{len(eligible_ids)} out of {credit_df['ID'].nunique()}")

credit_df_eligible = credit_df[credit_df['ID'].isin(eligible_ids)]
target_df = credit_df_eligible.groupby('ID')['is_risky'].max().reset_index()
target_df['TARGET'] = 1 - target_df['is_risky']  # 1=Approved/good, 0=Rejected/risky
target_df = target_df.drop(columns=['is_risky'])

print(f"Label distribution (vintage-filtered):")
print(target_df['TARGET'].value_counts(normalize=True))

# ----------------------------------------------------------------------
# Merge Applicant Data with Target
# ----------------------------------------------------------------------
df = app_df.merge(target_df, on='ID', how='inner')
print(f"After merge: {df.shape}")

# ----------------------------------------------------------------------
# Handle Missing Values (OCCUPATION_TYPE) -- fill, don't drop (V2 fix)
# ----------------------------------------------------------------------
df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')

# ----------------------------------------------------------------------
# Fix DAYS_BIRTH / DAYS_EMPLOYED
# ----------------------------------------------------------------------
df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).astype(int)
df['NOT_EMPLOYED_FLAG'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = 0
df['EMPLOYED_YEARS'] = (-df['DAYS_EMPLOYED'] / 365).clip(lower=0)
df = df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED'])

# ----------------------------------------------------------------------
# Drop Non-Informative / Redundant Columns
# ----------------------------------------------------------------------
df = df.drop(columns=['FLAG_MOBIL'])
df['DEPENDENTS_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS'].replace(0, 1)
df['FAMILY_SIZE'] = df['CNT_FAM_MEMBERS']
df = df.drop(columns=['CNT_CHILDREN', 'CNT_FAM_MEMBERS'])

# ----------------------------------------------------------------------
# Map Binary Flags
# ----------------------------------------------------------------------
df['CODE_GENDER'] = df['CODE_GENDER'].map({'M': 1, 'F': 0})
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})

# ----------------------------------------------------------------------
# Encode Categorical Columns (V2 fix: one-hot for nominal, ordinal for education)
# ----------------------------------------------------------------------
nominal_cols = ['NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'OCCUPATION_TYPE', 'NAME_HOUSING_TYPE']
df = pd.get_dummies(df, columns=nominal_cols)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

edu_order = {
    'Lower secondary': 0,
    'Secondary / secondary special': 1,
    'Incomplete higher': 2,
    'Higher education': 3,
    'Academic degree': 4,
}
df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].map(edu_order)

# ----------------------------------------------------------------------
# Handle Outliers (AMT_INCOME_TOTAL)
# ----------------------------------------------------------------------
cap = df['AMT_INCOME_TOTAL'].quantile(0.99)
df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].clip(upper=cap)

# ----------------------------------------------------------------------
# Final Checks
# ----------------------------------------------------------------------
print(df.isnull().sum().sum())  # expect 0
print(df.shape)
print(df['TARGET'].value_counts(normalize=True))

# ----------------------------------------------------------------------
# Save Cleaned Dataset
# ----------------------------------------------------------------------
final_df = df.drop(columns=['ID'])
final_df.to_csv('cleaned_credit_data_v3_vintage.csv', index=False)
print("Saved cleaned_credit_data_v3_vintage.csv:", final_df.shape)
