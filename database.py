"""
Database Manager
Handles SQLite database operations for prediction history
"""

import sqlite3
import os
from datetime import datetime


class DatabaseManager:
    """
    Manages SQLite database for storing prediction history
    """
    
    def __init__(self, db_path='database/predictions.db'):
        """
        Initialize database connection
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create database directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self.init_database()
    
    def get_connection(self):
        """
        Get database connection
        
        Returns:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """
        Create database tables if they don't exist
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_preview TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                word_count INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON predictions(timestamp DESC)
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, text, prediction, confidence, word_count=0):
        """
        Save a prediction to database
        
        Args:
            text (str): Original news text
            prediction (str): Prediction result (Real/Fake)
            confidence (float): Confidence percentage
            word_count (int): Number of words in text
            
        Returns:
            int: ID of saved record
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create text preview (first 100 characters)
        text_preview = text[:100].replace('\n', ' ').strip() + '...' if len(text) > 100 else text
        
        cursor.execute('''
            INSERT INTO predictions (text_preview, prediction, confidence, word_count)
            VALUES (?, ?, ?, ?)
        ''', (text_preview, prediction, confidence, word_count))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return record_id
    
    def get_all_predictions(self, limit=50):
        """
        Get all prediction history
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of prediction records
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        predictions = []
        for row in rows:
            predictions.append({
                'id': row['id'],
                'text_preview': row['text_preview'],
                'prediction': row['prediction'],
                'confidence': row['confidence'],
                'word_count': row['word_count'],
                'timestamp': row['timestamp']
            })
        
        return predictions
    
    def get_statistics(self):
        """
        Get prediction statistics
        
        Returns:
            dict: Statistics about predictions
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) as total FROM predictions')
        total = cursor.fetchone()['total']
        
        # Real vs Fake count
        cursor.execute('''
            SELECT prediction, COUNT(*) as count 
            FROM predictions 
            GROUP BY prediction
        ''')
        distribution = {row['prediction']: row['count'] for row in cursor.fetchall()}
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) as avg_conf FROM predictions')
        avg_confidence = cursor.fetchone()['avg_conf'] or 0
        
        # Today's predictions
        cursor.execute('''
            SELECT COUNT(*) as today_count 
            FROM predictions 
            WHERE DATE(timestamp) = DATE('now')
        ''')
        today_count = cursor.fetchone()['today_count']
        
        conn.close()
        
        return {
            'total': total,
            'real_count': distribution.get('Real News', 0),
            'fake_count': distribution.get('Fake News', 0),
            'avg_confidence': avg_confidence,
            'today_count': today_count
        }
    
    def clear_history(self):
        """
        Delete all prediction records
        
        Returns:
            int: Number of deleted records
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as count FROM predictions')
        count = cursor.fetchone()['count']
        
        cursor.execute('DELETE FROM predictions')
        conn.commit()
        conn.close()
        
        return count
    
    def delete_prediction(self, record_id):
        """
        Delete a specific prediction
        
        Args:
            record_id (int): ID of record to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM predictions WHERE id = ?', (record_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted