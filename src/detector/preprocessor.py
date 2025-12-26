"""Data preprocessing and feature engineering"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """Preprocess network flows"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def fit(self, X: pd.DataFrame):
        """Learn preprocessing parameters"""
        self.feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.scaler.fit(X[self.feature_columns])
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply preprocessing"""
        X_numeric = X[self.feature_columns].fillna(0)
        return self.scaler.transform(X_numeric)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform"""
        return self.fit(X).transform(X)
