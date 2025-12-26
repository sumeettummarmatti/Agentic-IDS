"""
Feature preprocessing with MULTI-LABEL classification
Matches your notebook approach: detect specific attack types, not just benign/attack
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Handles data loading and preprocessing from CIC-Darknet2020
    
    TWO-LEVEL APPROACH:
    1. Binary level: Encryption (benign vs attack-like) for quick filtering
    2. Multi-label level: Label (specific attack type) for detailed classification
    """
    
    # Features to drop (non-predictive)
    DROP_FEATURES = [
        'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 
        'Src Port', 'Dst Port', 'Flow Duration',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
        'Type'  # We'll use 'Label' instead
    ]
    
    # ALL attack types in the dataset
    ATTACK_CLASSES = [
        'AUDIO-STREAMING',
        'VIDEO-STREAMING', 
        'VOIP',
        'BROWSING',
        'FILE-TRANSFER',
        'P2P',
        'EMAIL',
        'CHAT',
        'STREAMING',
        'TORRENT'
    ]
    
    def __init__(self, use_multi_label=True):
        """
        Initialize preprocessor
        
        Args:
            use_multi_label: If True, classify to specific attack types
                           If False, binary benign/attack classification
        """
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.use_multi_label = use_multi_label
    
    def load_data(self, filepath):
        """Load CIC-Darknet2020 dataset"""
        logger.info(f"Loading data from {filepath}...")
        
        df = pd.read_excel(filepath)
        logger.info(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Drop irrelevant columns
        cols_to_drop = [c for c in self.DROP_FEATURES if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # Remove Unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Drop NaN
        df = df.dropna()
        logger.info(f"✓ After cleaning: {len(df)} rows")
        
        return df
    
    def prepare_features_and_labels(self, df):
        """
        Prepare features and labels with TWO CLASSIFICATION SCHEMES
        
        Returns:
            tuple: (X, y_binary, y_multi) where:
            - X: Features (71 numeric columns)
            - y_binary: Binary labels (0=benign, 1=attack-like) from Encryption
            - y_multi: Multi-class labels (specific attack types) from Label
        """
        
        # ====== BINARY LEVEL ======
        # Encryption: Is this benign or attack-like?
        if 'Type' in df.columns:
            df['Encryption'] = df['Type'].map({
                'Non-Tor': 0,    # Benign
                'NonVPN': 0,     # Benign
                'Tor': 1,        # Attack-like (encrypted anonymity)
                'VPN': 1         # Attack-like (encrypted anonymity)
            })
        
        y_binary = df['Encryption'].values if 'Encryption' in df.columns else None
        
        # ====== MULTI-LABEL LEVEL ======
        # Label: What specific type of attack/traffic is this?
        if 'Label' in df.columns:
            # Encode the specific attack types
            self.label_encoder.fit(self.ATTACK_CLASSES)
            y_multi = self.label_encoder.transform(df['Label'])
        else:
            y_multi = None
        
        # ====== FEATURES ======
        # Select numeric features only
        X = df.select_dtypes(include=[np.number])
        X = X.fillna(0)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        logger.info(f"Features: {len(self.feature_names)}")
        
        if y_binary is not None:
            logger.info(f"Binary distribution: {np.bincount(y_binary.astype(int))}")
        if y_multi is not None:
            logger.info(f"Multi-class distribution: {np.bincount(y_multi.astype(int))}")
        
        return X.values, y_binary, y_multi
    
    def fit_scaler(self, X):
        """Fit scaler on training data"""
        self.scaler.fit(X)
        logger.info("✓ Scaler fitted")
    
    def transform(self, X):
        """Transform data using fitted scaler"""
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.scaler.fit_transform(X)
    
    def get_label_name(self, label_index):
        """Convert label index back to attack type name"""
        if hasattr(self, 'label_encoder'):
            return self.label_encoder.inverse_transform([label_index])
        return "Unknown"
