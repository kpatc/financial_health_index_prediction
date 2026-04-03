"""
Model training pipeline for Financial Health Index prediction.
Includes preprocessing, model training, cross-validation, and evaluation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def build_preprocessor(X_train):
    """
    Build preprocessing pipeline using ColumnTransformer.

    Numeric features: impute (median) → scale
    Categorical features: impute (mode) → ordinal encode
    """
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove ID if present
    if 'ID' in numeric_features:
        numeric_features.remove('ID')
    if 'ID' in categorical_features:
        categorical_features.remove('ID')

    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor, numeric_features, categorical_features


def train_and_evaluate_models(X_train, y_train, preprocessor, numeric_features, categorical_features):
    """
    Train multiple models and evaluate with StratifiedKFold cross-validation.
    Returns results dictionary and trained models.
    """
    results = {}
    trained_models = {}

    # Use stratified k-fold to handle class imbalance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "="*70)
    print("MODEL TRAINING & CROSS-VALIDATION")
    print("="*70)

    # Model 1: Logistic Regression (baseline)
    print("\n[1/4] Logistic Regression (class_weight='balanced')...")
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    lr_cv_results = cross_validate(
        lr_pipeline, X_train, y_train, cv=skf,
        scoring=['accuracy', 'f1_weighted', 'f1_macro'],
        return_train_score=True
    )
    results['Logistic Regression'] = {
        'train_f1_weighted': lr_cv_results['train_f1_weighted'].mean(),
        'test_f1_weighted': lr_cv_results['test_f1_weighted'].mean(),
        'train_f1_macro': lr_cv_results['train_f1_macro'].mean(),
        'test_f1_macro': lr_cv_results['test_f1_macro'].mean(),
        'train_accuracy': lr_cv_results['train_accuracy'].mean(),
        'test_accuracy': lr_cv_results['test_accuracy'].mean(),
    }
    lr_pipeline.fit(X_train, y_train)
    trained_models['Logistic Regression'] = lr_pipeline
    print(f"  ✓ F1 (weighted): {results['Logistic Regression']['test_f1_weighted']:.4f}")
    print(f"  ✓ F1 (macro):    {results['Logistic Regression']['test_f1_macro']:.4f}")
    print(f"  ✓ Accuracy:      {results['Logistic Regression']['test_accuracy']:.4f}")

    # Model 2: Random Forest
    print("\n[2/4] Random Forest (class_weight='balanced')...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    rf_cv_results = cross_validate(
        rf_pipeline, X_train, y_train, cv=skf,
        scoring=['accuracy', 'f1_weighted', 'f1_macro'],
        return_train_score=True
    )
    results['Random Forest'] = {
        'train_f1_weighted': rf_cv_results['train_f1_weighted'].mean(),
        'test_f1_weighted': rf_cv_results['test_f1_weighted'].mean(),
        'train_f1_macro': rf_cv_results['train_f1_macro'].mean(),
        'test_f1_macro': rf_cv_results['test_f1_macro'].mean(),
        'train_accuracy': rf_cv_results['train_accuracy'].mean(),
        'test_accuracy': rf_cv_results['test_accuracy'].mean(),
    }
    rf_pipeline.fit(X_train, y_train)
    trained_models['Random Forest'] = rf_pipeline
    print(f"  ✓ F1 (weighted): {results['Random Forest']['test_f1_weighted']:.4f}")
    print(f"  ✓ F1 (macro):    {results['Random Forest']['test_f1_macro']:.4f}")
    print(f"  ✓ Accuracy:      {results['Random Forest']['test_accuracy']:.4f}")

    # Model 3: Gradient Boosting
    print("\n[3/4] Gradient Boosting...")
    gb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        ))
    ])

    gb_cv_results = cross_validate(
        gb_pipeline, X_train, y_train, cv=skf,
        scoring=['accuracy', 'f1_weighted', 'f1_macro'],
        return_train_score=True
    )
    results['Gradient Boosting'] = {
        'train_f1_weighted': gb_cv_results['train_f1_weighted'].mean(),
        'test_f1_weighted': gb_cv_results['test_f1_weighted'].mean(),
        'train_f1_macro': gb_cv_results['train_f1_macro'].mean(),
        'test_f1_macro': gb_cv_results['test_f1_macro'].mean(),
        'train_accuracy': gb_cv_results['train_accuracy'].mean(),
        'test_accuracy': gb_cv_results['test_accuracy'].mean(),
    }
    gb_pipeline.fit(X_train, y_train)
    trained_models['Gradient Boosting'] = gb_pipeline
    print(f"  ✓ F1 (weighted): {results['Gradient Boosting']['test_f1_weighted']:.4f}")
    print(f"  ✓ F1 (macro):    {results['Gradient Boosting']['test_f1_macro']:.4f}")
    print(f"  ✓ Accuracy:      {results['Gradient Boosting']['test_accuracy']:.4f}")

    # Model 4: XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n[4/4] XGBoost...")
        xgb_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ))
        ])

        xgb_cv_results = cross_validate(
            xgb_pipeline, X_train, y_train, cv=skf,
            scoring=['accuracy', 'f1_weighted', 'f1_macro'],
            return_train_score=True
        )
        results['XGBoost'] = {
            'train_f1_weighted': xgb_cv_results['train_f1_weighted'].mean(),
            'test_f1_weighted': xgb_cv_results['test_f1_weighted'].mean(),
            'train_f1_macro': xgb_cv_results['train_f1_macro'].mean(),
            'test_f1_macro': xgb_cv_results['test_f1_macro'].mean(),
            'train_accuracy': xgb_cv_results['train_accuracy'].mean(),
            'test_accuracy': xgb_cv_results['test_accuracy'].mean(),
        }
        xgb_pipeline.fit(X_train, y_train)
        trained_models['XGBoost'] = xgb_pipeline
        print(f"  ✓ F1 (weighted): {results['XGBoost']['test_f1_weighted']:.4f}")
        print(f"  ✓ F1 (macro):    {results['XGBoost']['test_f1_macro']:.4f}")
        print(f"  ✓ Accuracy:      {results['XGBoost']['test_accuracy']:.4f}")
    else:
        print("\n[4/4] XGBoost... SKIPPED (not installed)")

    return results, trained_models


def print_results_summary(results):
    """Print model comparison summary."""
    print("\n" + "="*70)
    print("MODEL COMPARISON (Cross-Validation Results)")
    print("="*70)

    results_df = pd.DataFrame(results).T
    print(results_df.to_string())

    # Find best model
    best_model = results_df['test_f1_weighted'].idxmax()
    best_score = results_df.loc[best_model, 'test_f1_weighted']

    print(f"\n✓ BEST MODEL: {best_model} (F1 weighted = {best_score:.4f})")

    return best_model


def get_feature_importance(trained_model, numeric_features, categorical_features, top_n=20):
    """
    Extract feature importance from tree-based models.
    """
    try:
        classifier = trained_model.named_steps['classifier']

        # For tree-based models
        if hasattr(classifier, 'feature_importances_'):
            feature_names = numeric_features + categorical_features
            importances = classifier.feature_importances_

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            return importance_df.head(top_n)
    except:
        pass

    return None


def save_artifacts(best_pipeline, preprocessor, model_name, output_dir):
    """Save model and preprocessor artifacts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_path = Path(output_dir) / 'model.pkl'
    preprocessor_path = Path(output_dir) / 'preprocessor.pkl'

    joblib.dump(best_pipeline, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Preprocessor saved: {preprocessor_path}")


if __name__ == '__main__':
    # Load cleaned data
    print("Loading cleaned training data...")
    from cleaning.clean import load_data, clean_pipeline, split_features_target

    train, test = load_data('data')
    train_clean, test_clean = clean_pipeline(train, test)

    X_train, y_train = split_features_target(train_clean)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")

    # Build preprocessor
    print("\nBuilding preprocessing pipeline...")
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    print(f"Numeric features ({len(numeric_features)}): {numeric_features[:5]}...")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features[:5]}...")

    # Train and evaluate models
    results, trained_models = train_and_evaluate_models(
        X_train, y_train, preprocessor, numeric_features, categorical_features
    )

    # Print summary
    best_model_name = print_results_summary(results)
    best_pipeline = trained_models[best_model_name]

    # Feature importance
    importance_df = get_feature_importance(best_pipeline, numeric_features, categorical_features)
    if importance_df is not None:
        print(f"\n✓ Top 20 Features by Importance ({best_model_name}):")
        print(importance_df.to_string(index=False))

    # Save artifacts
    save_artifacts(best_pipeline, preprocessor, best_model_name, 'modeling')

    print("\n✓ Training complete!")
