"""
Utility Functions
Helper functions for text processing and file operations
"""

import os
import re
import string
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


def clean_text(text):
    """
    Clean and preprocess text data
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words])
    
    # Stemming
    stemmer = PorterStemmer()
    words = text.split()
    text = ' '.join([stemmer.stem(word) for word in words])
    
    return text


def extract_features(text):
    """
    Extract additional text features for analysis
    
    Args:
        text (str): Raw text
        
    Returns:
        dict: Dictionary of extracted features
    """
    features = {
        'char_count': len(text),
        'word_count': len(text.split()),
        'avg_word_length': 0,
        'unique_word_ratio': 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': 0,
        'has_breaking': 1 if 'breaking' in text.lower() else 0,
        'has_exclusive': 1 if 'exclusive' in text.lower() else 0,
        'has_shocking': 1 if 'shocking' in text.lower() else 0,
        'has_urgent': 1 if 'urgent' in text.lower() else 0
    }
    
    # Calculate ratios
    words = text.split()
    if len(words) > 0:
        features['avg_word_length'] = sum(len(w) for w in words) / len(words)
        unique_words = set(words)
        features['unique_word_ratio'] = len(unique_words) / len(words)
    
    # Uppercase ratio
    if len(text) > 0:
        uppercase_count = sum(1 for c in text if c.isupper())
        features['uppercase_ratio'] = uppercase_count / len(text)
    
    return features


def load_model_and_vectorizer():
    """
    Load trained model and vectorizer
    
    Returns:
        tuple: (model, vectorizer) or (None, None) if not found
    """
    model_path = 'model/fake_news_model.pkl'
    vectorizer_path = 'model/tfidf_vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    
    return None, None


def save_model_and_vectorizer(model, vectorizer):
    """
    Save trained model and vectorizer
    
    Args:
        model: Trained ML model
        vectorizer: Fitted TF-IDF vectorizer
    """
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/fake_news_model.pkl')
    joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')


def get_model_info():
    """
    Get information about the trained model
    
    Returns:
        dict: Model information
    """
    info = {
        'model_type': 'Logistic Regression',
        'vectorizer': 'TF-IDF (Term Frequency-Inverse Document Frequency)',
        'preprocessing': ['Lowercase', 'URL Removal', 'HTML Tag Removal', 
                         'Stopword Removal', 'Stemming'],
        'features': '5000 most frequent terms',
        'training_data': 'Real-world news articles dataset'
    }
    
    model_path = 'model/fake_news_model.pkl'
    if os.path.exists(model_path):
        info['status'] = 'Trained'
        info['model_size'] = f"{os.path.getsize(model_path) / 1024:.1f} KB"
    else:
        info['status'] = 'Not trained yet'
    
    return info


def generate_sample_text():
    """
    Generate sample news texts for demonstration
    
    Returns:
        dict: Dictionary of sample real and fake news examples
    """
    samples = {
        "real_news": """The United Nations Climate Change Conference (COP28) concluded 
        today with world leaders agreeing to new emissions reduction targets. The 
        agreement aims to limit global temperature rise to 1.5 degrees Celsius above 
        pre-industrial levels. Representatives from over 190 countries participated 
        in the two-week summit held in Dubai, United Arab Emirates. The final 
        declaration calls for a 43% reduction in greenhouse gas emissions by 2030.""",
        
        "fake_news": """BREAKING: Scientists discover that the Earth is actually 
        hollow and inhabited by ancient civilizations! SHOCKING evidence revealed 
        by anonymous government insider. This INCREDIBLE discovery will change 
        everything we know about our planet. The secret has been hidden for 
        centuries but now the TRUTH is finally coming out. Click SHARE to spread 
        this AMAZING news before it's CENSORED!!!"""
    }
    return samples


def analyze_news_article(text):
    """
    Analyze news article and return detailed metrics
    
    Args:
        text (str): News article text
        
    Returns:
        dict: Analysis results
    """
    features = extract_features(text)
    cleaned = clean_text(text)
    
    analysis = {
        'word_count': features['word_count'],
        'char_count': features['char_count'],
        'avg_word_length': round(features['avg_word_length'], 2),
        'unique_word_ratio': round(features['unique_word_ratio'], 2),
        'exclamation_count': features['exclamation_count'],
        'question_count': features['question_count'],
        'uppercase_ratio': round(features['uppercase_ratio'], 3),
        'sensational_words': {
            'breaking': features['has_breaking'],
            'exclusive': features['has_exclusive'],
            'shocking': features['has_shocking'],
            'urgent': features['has_urgent']
        },
        'cleaned_length': len(cleaned),
        'suspicious_score': calculate_suspicious_score(features)
    }
    
    return analysis


def calculate_suspicious_score(features):
    """
    Calculate a suspicious score based on text features
    
    Args:
        features (dict): Extracted text features
        
    Returns:
        float: Suspicious score between 0 and 1
    """
    score = 0.0
    
    # Sensational words increase suspicious score
    if features['has_breaking']:
        score += 0.15
    if features['has_exclusive']:
        score += 0.1
    if features['has_shocking']:
        score += 0.2
    if features['has_urgent']:
        score += 0.15
    
    # High exclamation/question marks
    if features['exclamation_count'] > 3:
        score += 0.15
    if features['question_count'] > 5:
        score += 0.1
    
    # High uppercase ratio
    if features['uppercase_ratio'] > 0.1:
        score += 0.15
    
    return min(score, 1.0)