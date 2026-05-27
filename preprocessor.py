"""
Text Preprocessor
Handles text cleaning and feature extraction for model training
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils import clean_text


def load_and_prepare_data(train_path='data/train.csv'):
    """
    Load dataset and prepare for training
    """
    print("Loading dataset...")
    
    # Create sample dataset if file doesn't exist
    try:
        df = pd.read_csv(train_path)
        print(f"Loaded {len(df)} records from {train_path}")
    except FileNotFoundError:
        print("Dataset not found. Creating sample data...")
        df = create_sample_dataset()
    
    # Combine title and text for better feature extraction
    if 'title' in df.columns and 'text' in df.columns:
        df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'text' in df.columns:
        df['full_text'] = df['text'].fillna('')
    else:
        df['full_text'] = df.iloc[:, 0].astype(str)
    
    # Clean text
    print("Cleaning text data...")
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 10].copy()
    
    # ============ FIX: Convert to Python lists ============
    X = df['cleaned_text'].tolist()  # Convert to list
    y = df['label'].tolist()         # Convert to list
    
    print(f"Total samples: {len(X)}")
    print(f"Fake news samples: {sum(y)}")
    print(f"Real news samples: {len(y) - sum(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def create_vectorizer(max_features=5000):
    """
    Create and return a TF-IDF vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        strip_accents='unicode'
    )
    
    return vectorizer


def create_sample_dataset():
    """
    Create a sample dataset for demonstration
    """
    real_news = [
        "Scientists have discovered a new method for treating diabetes that could help millions of patients worldwide.",
        "The United Nations General Assembly passed a resolution on climate change with overwhelming support from member states.",
        "Local community members raised over fifty thousand dollars for school renovation projects in the downtown area.",
        "The city council approved a new public transportation plan that will add twenty new bus routes by next year.",
        "A comprehensive study published in a medical journal shows significant benefits of regular exercise for heart health.",
        "Technology company announced a new renewable energy initiative that will power thousands of homes with solar energy.",
        "Research team from the university published their findings on ocean conservation efforts in a peer-reviewed journal.",
        "Government officials released the quarterly economic growth report showing steady improvement in key sectors.",
        "The state university launched a new scholarship program that will help hundreds of students afford higher education.",
        "Healthcare workers received special recognition for their dedicated service during the recent health crisis.",
        "International space station crew completed a successful spacewalk to repair critical equipment on the station.",
        "World health organization released new guidelines for maintaining mental health during challenging times.",
        "The supreme court issued a landmark ruling that will affect environmental protection policies across the nation.",
        "Major automaker announced plans to transition completely to electric vehicles by the next decade.",
        "Public library system expanded its digital resources making thousands of ebooks available for free to residents."
    ]
    
    fake_news = [
        "BREAKING: Secret government plot to control the weather revealed by anonymous whistleblower! SHOCKING details!",
        "You will not BELIEVE what this celebrity did to fake their own death for insurance money! EXCLUSIVE report!",
        "Scientists SHOCKED by this one weird trick that cures all diseases! Doctors HATE this! MUST READ NOW!!!",
        "URGENT: Massive conspiracy uncovered in the food industry! They are POISONING your children! Share this NOW!",
        "Anonymous source reveals that NASA has been hiding evidence of alien contact for DECADES! The TRUTH is out!",
        "This one simple trick will make you a MILLIONAIRE overnight! Banks are FURIOUS about this secret!",
        "SHOCKING: Government secretly putting microchips in drinking water to control the population! WAKE UP!",
        "EXCLUSIVE: President caught in scandal that changes EVERYTHING! You will not believe what happens next!",
        "BREAKING NEWS: Famous scientist admits that the Earth is actually FLAT! The cover-up is OVER!!!",
        "This miracle supplement CURES everything! Pharmaceutical companies are trying to BAN it! Buy NOW!",
        "ALERT: Your smartphone is SPYING on you through a secret app! Here is how to remove it immediately!",
        "The TRUTH about vaccines that they do not want you to know! This changes EVERYTHING! Share before deleted!",
        "OMG: Popular food item contains DEADLY ingredients! The FDA is covering it up! Boycott NOW!!!",
        "SECRET REVEALED: Banks are stealing your money through hidden fees! Here is how to protect yourself!",
        "You will not believe what the government is hiding about the moon landing! The evidence is SHOCKING!"
    ]
    
    texts = []
    labels = []
    
    # Add real news (repeat 4 times for more data)
    for text in real_news:
        for _ in range(4):
            texts.append(text)
            labels.append(0)
    
    # Add fake news (repeat 4 times for more data)
    for text in fake_news:
        for _ in range(4):
            texts.append(text)
            labels.append(1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Created dataset with {len(df)} records")
    print(f"  - Real news: {len(df) - df['label'].sum()}")
    print(f"  - Fake news: {df['label'].sum()}")
    
    return df