"""
Machine Learning Model
Trains and manages the fake news detection model
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from preprocessor import load_and_prepare_data, create_vectorizer
from utils import save_model_and_vectorizer


def train_model(X_train, X_test, y_train, y_test, model_type='logistic'):
    """
    Train a machine learning model for fake news detection
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_type (str): Type of model to train
        
    Returns:
        tuple: (trained_model, vectorizer, metrics_dict)
    """
    print(f"\nTraining {model_type} model...")
    
    # Create and fit vectorizer
    vectorizer = create_vectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Select model
    models = {
        'logistic': LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'naive_bayes': MultinomialNB(alpha=0.1),
        'svm': LinearSVC(
            C=1.0,
            max_iter=1000,
            random_state=42,
            dual=False
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    }
    
    if model_type not in models:
        print(f"Unknown model type: {model_type}. Using Logistic Regression.")
        model_type = 'logistic'
    
    model = models[model_type]
    
    # Train model
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_tfidf)
    y_pred_test = model.predict(X_test_tfidf)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, n_jobs=-1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    
    # Feature importance (if available)
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        top_indices = importance.argsort()[-10:][::-1]
        top_features = [(feature_names[i], importance[i]) for i in top_indices]
    else:
        top_features = []
    
    metrics = {
        'model_type': model_type.replace('_', ' ').title(),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
        'top_features': top_features,
        'vectorizer_features': len(feature_names)
    }
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model, vectorizer, metrics


def predict_news(text, model, vectorizer):
    """
    Predict whether a news article is real or fake
    
    Args:
        text (str): News article text
        model: Trained ML model
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        tuple: (prediction_label, confidence_score)
    """
    from utils import clean_text
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Transform to TF-IDF features
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_tfidf)[0]
        prediction = model.predict(text_tfidf)[0]
        confidence = max(probabilities) * 100
    else:
        # For models without predict_proba (like SVM)
        prediction = model.predict(text_tfidf)[0]
        decision_score = model.decision_function(text_tfidf)[0]
        # Convert decision score to confidence-like score
        confidence = 1 / (1 + np.exp(-abs(decision_score))) * 100
    
    label = "Fake News" if prediction == 1 else "Real News"
    
    return label, confidence


def compare_models(X_train, X_test, y_train, y_test):
    """
    Compare multiple models to find the best one
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        pd.DataFrame: Comparison results
    """
    model_types = ['logistic', 'naive_bayes', 'svm', 'random_forest']
    results = []
    
    for model_type in model_types:
        try:
            model, vectorizer, metrics = train_model(
                X_train, X_test, y_train, y_test, model_type
            )
            results.append({
                'Model': model_type.title().replace('_', ' '),
                'Train Accuracy': metrics['train_accuracy'],
                'Test Accuracy': metrics['test_accuracy'],
                'CV Mean': metrics['cv_mean'],
                'CV Std': metrics['cv_std']
            })
        except Exception as e:
            print(f"Error with {model_type}: {e}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test Accuracy', ascending=False)
    
    return results_df