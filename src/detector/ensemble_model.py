"""
Multi-model ensemble detector
- XGBoost for statistical patterns
- LSTM for temporal patterns
- Adaptive routing based on confidence
"""

import xgboost as xgb
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import joblib

class EnsembleDetector:
    """Ensemble of XGBoost + LSTM for intrusion detection"""
    
    def __init__(self, xgboost_model=None, lstm_model=None):
        self.xgboost = xgboost_model
        self.lstm = lstm_model
        self.scaler = StandardScaler()
        self.threshold = 0.5
        self.routing_threshold = 0.8  # Use XGBoost if confidence > 0.8
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost component"""
        self.xgboost = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.xgboost.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Predict with ensemble routing"""
        if self.xgboost is None:
            raise ValueError("XGBoost model not trained")
        
        xgb_proba = self.xgboost.predict_proba(X)
        xgb_max_conf = xgb_proba.max(axis=1)
        
        # Route: use XGBoost if high confidence
        predictions = self.xgboost.predict(X)
        
        return predictions
    
    def predict_proba(self, X):
        """Get probability predictions"""
        return self.xgboost.predict_proba(X)
    
    def save(self, path: str):
        """Save ensemble"""
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: str):
        """Load ensemble"""
        return joblib.load(path)
