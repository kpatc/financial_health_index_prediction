"""
Generate predictions on test set using trained model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib


def load_model_and_preprocessor(model_dir='modeling'):
    """Load saved model and preprocessor."""
    model_path = Path(model_dir) / 'model.pkl'
    preprocessor_path = Path(model_dir) / 'preprocessor.pkl'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    print(f"✓ Loaded model from {model_path}")
    print(f"✓ Loaded preprocessor from {preprocessor_path}")

    return model, preprocessor


def generate_predictions(X_test, model, test_ids=None):
    """
    Generate class predictions and probability predictions.

    Returns:
    - predictions: class labels (0=Low, 1=Medium, 2=High)
    - probabilities: probability matrix (n_samples × 3)
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    return predictions, probabilities


def save_predictions_submission(test_ids, predictions, probabilities, output_file='test_predictions.csv'):
    """
    Save predictions in submission format.
    Includes: ID, prediction class, and probability for each class.
    """
    class_names = ['Low', 'Medium', 'High']

    submission_df = pd.DataFrame({
        'ID': test_ids,
        'Prediction': [class_names[p] for p in predictions],
        'Prob_Low': probabilities[:, 0],
        'Prob_Medium': probabilities[:, 1],
        'Prob_High': probabilities[:, 2]
    })

    submission_df.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to {output_file}")

    return submission_df


if __name__ == '__main__':
    # Load cleaned test data
    print("Loading test data...")
    from cleaning.clean import load_data, clean_pipeline, prepare_test_features

    train, test = load_data('data')
    train_clean, test_clean = clean_pipeline(train, test)

    X_test = prepare_test_features(test_clean)
    test_ids = test['ID']

    print(f"X_test shape: {X_test.shape}")

    # Load trained model
    print("\nLoading trained model...")
    model, preprocessor = load_model_and_preprocessor()

    # Generate predictions
    print("\nGenerating predictions...")
    predictions, probabilities = generate_predictions(X_test, model, test_ids)

    # Save submissions
    submission_df = save_predictions_submission(test_ids, predictions, probabilities)

    print("\nPrediction summary:")
    class_names = ['Low', 'Medium', 'High']
    for i, class_name in enumerate(class_names):
        count = (predictions == i).sum()
        pct = count / len(predictions) * 100
        print(f"  {class_name}: {count:4d} ({pct:5.1f}%)")

    print(f"\nFirst 5 predictions:")
    print(submission_df.head())
    print("\n✓ Prediction complete!")
