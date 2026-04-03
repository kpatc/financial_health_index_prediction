"""
Data cleaning and preprocessing module for Financial Health Index dataset.
Handles string normalization, encoding, log transforms, and feature engineering.
"""
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path


def normalize_strings(df):
    """
    Normalize categorical string values:
    - Remove/normalize unicode characters (e.g., zero-width spaces, smart quotes)
    - Standardize apostrophes
    - Strip whitespace
    """
    df = df.copy()

    # Get all string/object columns
    string_cols = df.select_dtypes(include=['object']).columns

    for col in string_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: _normalize_string(x) if pd.notna(x) else x)

    return df


def _normalize_string(s):
    """Helper: normalize a single string value."""
    if not isinstance(s, str):
        return s

    # Remove unicode control/format characters (zero-width spaces, etc)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Cf')

    # Fix common corruptions: "Don?t" → "Don't", curly quotes → straight
    s = s.replace(''', "'").replace(''', "'")
    s = s.replace('"', '"').replace('"', '"')
    s = re.sub(r'Don\?t', "Don't", s)
    s = re.sub(r'doesn\?t', "doesn't", s)

    # Strip whitespace
    s = s.strip()

    return s


def encode_ordinal_features(df):
    """
    Encode 3-tier ownership/access features as ordinal:
    Never had / Used to have but don't have now / Have now → 0, 1, 2
    Keep NaN as NaN.
    """
    df = df.copy()

    three_tier_cols = [
        'motor_vehicle_insurance', 'has_mobile_money', 'has_credit_card',
        'has_loan_account', 'has_internet_banking', 'has_debit_card',
        'medical_insurance', 'funeral_insurance',
        'uses_friends_family_savings', 'uses_informal_lender'
    ]

    ordinal_map = {
        'Never had': 0,
        'Used to have but don\'t have now': 1,
        'Have now': 2,
    }

    for col in three_tier_cols:
        if col in df.columns:
            df[col] = df[col].map(ordinal_map)

    return df


def encode_binary_features(df):
    """
    Encode binary Yes/No features:
    Yes → 1, No → 0, Don't know/NaN → NaN (or could map to 0.5)
    """
    df = df.copy()

    binary_cols = [
        'attitude_stable_business_environment',
        'attitude_worried_shutdown',
        'compliance_income_tax',
        'perception_insurance_doesnt_cover_losses',
        'perception_cannot_afford_insurance',
        'current_problem_cash_flow',
        'has_cellphone',
        'offers_credit_to_customers',
        'attitude_satisfied_with_achievement',
        'perception_insurance_companies_dont_insure_businesses_like_yours',
        'perception_insurance_important',
        'has_insurance',
        'covid_essential_service',
        'attitude_more_successful_next_year',
        'problem_sourcing_money',
        'marketing_word_of_mouth',
        'future_risk_theft_stock',
        'motivation_make_more_money'
    ]

    binary_map = {
        'Yes': 1,
        'No': 0,
    }

    for col in binary_cols:
        if col in df.columns:
            # Map Yes/No, leave Don't know/NaN as NaN
            df[col] = df[col].apply(
                lambda x: binary_map.get(x, np.nan) if pd.notna(x) else np.nan
            )
            df[col] = df[col].astype('float64')

    return df


def encode_categorical_features(df):
    """
    Encode remaining categorical features (country, owner_sex, keeps_financial_records).
    """
    df = df.copy()

    # Country: simple integer encoding
    if 'country' in df.columns:
        country_map = {
            'eswatini': 0,
            'lesotho': 1,
            'malawi': 2,
            'zimbabwe': 3,
        }
        df['country'] = df['country'].str.lower().map(country_map)

    # Owner sex: Female=0, Male=1
    if 'owner_sex' in df.columns:
        sex_map = {
            'Female': 0,
            'Male': 1,
        }
        df['owner_sex'] = df['owner_sex'].map(sex_map)

    # keeps_financial_records: Yes/Always=1, Yes/Sometimes=0.5, No=0
    if 'keeps_financial_records' in df.columns:
        def encode_financial_records(x):
            if pd.isna(x):
                return np.nan
            x = str(x).lower()
            if 'no' in x:
                return 0.0
            elif 'sometimes' in x:
                return 0.5
            elif 'yes' in x or 'always' in x:
                return 1.0
            return np.nan

        df['keeps_financial_records'] = df['keeps_financial_records'].apply(encode_financial_records)

    return df


def log_transform_financial(df):
    """
    Log-transform financial features due to extreme right skew.
    Use log(x + 1) to handle zeros.
    """
    df = df.copy()

    financial_cols = [
        'personal_income',
        'business_expenses',
        'business_turnover'
    ]

    for col in financial_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])

    return df


def add_engineered_features(df):
    """
    Create derived features:
    - combined_business_age: years + months/12
    - income_expense_ratio: log(income+1) / log(expenses+1)
    - expense_turnover_ratio: log(expenses+1) / log(turnover+1)
    - insurance_product_count: sum of insurance holdings
    - formal_finance_count: sum of formal financial product holdings
    """
    df = df.copy()

    # Combined business age
    if 'business_age_years' in df.columns and 'business_age_months' in df.columns:
        df['combined_business_age'] = (
            df['business_age_years'].fillna(0) + df['business_age_months'].fillna(0) / 12
        )

    # Income to expense ratio (only where both exist)
    if 'personal_income_log' in df.columns and 'business_expenses_log' in df.columns:
        df['income_expense_ratio'] = (
            df['personal_income_log'] - df['business_expenses_log']
        )
        df['income_expense_ratio'] = df['income_expense_ratio'].fillna(0)

    # Expense to turnover ratio
    if 'business_expenses_log' in df.columns and 'business_turnover_log' in df.columns:
        df['expense_turnover_ratio'] = (
            df['business_expenses_log'] - df['business_turnover_log']
        )
        df['expense_turnover_ratio'] = df['expense_turnover_ratio'].fillna(0)

    # Count financial insurance products (ordinal encoded as 0/1/2)
    insurance_cols = ['motor_vehicle_insurance', 'medical_insurance', 'funeral_insurance']
    insurance_cols = [c for c in insurance_cols if c in df.columns]
    if insurance_cols:
        df['insurance_product_count'] = df[insurance_cols].apply(
            lambda row: row.sum() if row.notna().any() else 0, axis=1
        )

    # Count formal financial products (ordinal: 0/1/2)
    formal_cols = [
        'has_loan_account', 'has_credit_card',
        'has_internet_banking', 'has_debit_card'
    ]
    formal_cols = [c for c in formal_cols if c in df.columns]
    if formal_cols:
        df['formal_finance_count'] = df[formal_cols].apply(
            lambda row: row.sum() if row.notna().any() else 0, axis=1
        )

    return df


def encode_target(df):
    """Encode target: Low=0, Medium=1, High=2."""
    if 'Target' in df.columns:
        target_map = {'Low': 0, 'Medium': 1, 'High': 2}
        df['Target'] = df['Target'].map(target_map)
    return df


def load_data(data_dir):
    """Load Train.csv and Test.csv from data directory."""
    train_path = Path(data_dir) / 'Train.csv'
    test_path = Path(data_dir) / 'Test.csv'

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def clean_pipeline(train, test):
    """
    Apply full cleaning pipeline to train and test sets.
    Returns cleaned train and test DataFrames.
    """
    # Normalize strings first
    train = normalize_strings(train)
    test = normalize_strings(test)

    # Encode features
    train = encode_ordinal_features(train)
    test = encode_ordinal_features(test)

    train = encode_binary_features(train)
    test = encode_binary_features(test)

    train = encode_categorical_features(train)
    test = encode_categorical_features(test)

    # Log transforms
    train = log_transform_financial(train)
    test = log_transform_financial(test)

    # Feature engineering
    train = add_engineered_features(train)
    test = add_engineered_features(test)

    # Encode target (only for train)
    train = encode_target(train)

    return train, test


def split_features_target(train_df):
    """Split train data into X (features) and y (target)."""
    X = train_df.drop(columns=['ID', 'Target'])
    y = train_df['Target']
    return X, y


def prepare_test_features(test_df):
    """Extract test features (drop ID, no target)."""
    return test_df.drop(columns=['ID'])


def save_cleaned(train, test, output_dir):
    """Save cleaned train and test CSVs."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train.to_csv(Path(output_dir) / 'train_cleaned.csv', index=False)
    test.to_csv(Path(output_dir) / 'test_cleaned.csv', index=False)


if __name__ == '__main__':
    # Standalone execution: load, clean, save
    data_dir = '../data'
    output_dir = '../data/cleaned'

    print("Loading data...")
    train, test = load_data(data_dir)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    print("Cleaning pipeline...")
    train_clean, test_clean = clean_pipeline(train, test)
    print(f"Cleaned train shape: {train_clean.shape}, test shape: {test_clean.shape}")

    print("Saving cleaned data...")
    save_cleaned(train_clean, test_clean, output_dir)
    print(f"Saved to {output_dir}")

    print("\nSample of cleaned train:")
    print(train_clean.head())
    print(f"\nTarget distribution:\n{train_clean['Target'].value_counts(normalize=True)}")
