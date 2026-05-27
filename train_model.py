"""
Model Training Script
Run this script to train the fake news detection model
"""

import os
import sys
from preprocessor import load_and_prepare_data
from model import train_model, compare_models
from utils import save_model_and_vectorizer


def main():
    """
    Main training function
    """
    print("=" * 60)
    print("FAKE NEWS DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Create directories
    os.makedirs('model', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load and prepare data
    print("\n[1/4] Preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    if len(X_train) == 0:
        print("Error: No training data available!")
        sys.exit(1)
    
    # Compare models (optional)
    print("\n[2/4] Comparing models...")
    try:
        results = compare_models(X_train, X_test, y_train, y_test)
        print("\nModel Comparison:")
        print(results.to_string(index=False))
    except Exception as e:
        print(f"Model comparison skipped: {e}")
    
    # Train best model
    print("\n[3/4] Training final model...")
    model, vectorizer, metrics = train_model(
        X_train, X_test, y_train, y_test, 
        model_type='logistic'
    )
    
    # Save model
    print("\n[4/4] Saving model...")
    save_model_and_vectorizer(model, vectorizer)
    print("Model saved to 'model/' directory")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model Type: {metrics['model_type']}")
    print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Cross-validation: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    print(f"Features Used: {metrics['vectorizer_features']}")
    print("\nTop Predictive Features:")
    for feature, score in metrics['top_features'][:5]:
        print(f"  • {feature}: {score:.4f}")
    
    print("\n✅ Model training complete!")
    print("Run 'streamlit run app.py' to start the application.")


if __name__ == "__main__":
    main()