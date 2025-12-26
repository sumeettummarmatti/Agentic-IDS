"""
Feature preprocessing and engineering for CIC-Darknet2020
"""
#checkonce - get it checked from soham once - same settings as Xgboost file
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Handles data loading and preprocessing from CIC-Darknet2020
    """
    
    # Features to drop (non-predictive for attack detection) #checkonce
    DROP_FEATURES = [
        'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 
        'Src Port', 'Dst Port', 'Flow Duration',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
        'Type', 'Label'
    ]
    
    # Features for attack detection
    ATTACK_FEATURES = [
        'Protocol', 'Total Fwd Packet', 'Total Bwd packets',
        'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
        'Fwd Packet Length Max', 'Fwd Packet Length Min',
        'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Bwd Packet Length Max', 'Bwd Packet Length Min',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std',
        'Flow Bytes/s', 'Flow Packets/s',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
        'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
        'Fwd PSH Flags', 'Fwd SYN Flags', 'Fwd RST Flags',
        'Bwd PSH Flags', 'Bwd SYN Flags', 'Bwd RST Flags',
        'Fwd Init Win Bytes', 'Bwd Init Win Bytes',
    ]
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
    
    def load_data(self, filepath):  
        """
        Load CIC-Darknet2020 dataset
        
        Args:
            filepath: Path to Darknet.xlsx
            
        Returns:
            pd.DataFrame: Loaded and cleaned data
        """
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
        Prepare X and y from dataframe
        
        Args:
            df: Raw dataframe
            
        Returns:
            tuple: (X, y) ready for training
        """
        # Create encryption label (0=benign, 1=attack)
        # High encryption (Tor/VPN) = attack-like, Standard = benign
        df['Encryption'] = df['Type'].map({
            'Non-Tor': 0,
            'NonVPN': 0,
            'Tor': 1,
            'VPN': 1
        })
        
        y = df['Encryption'].values
        
        # Select numeric features
        X = df.select_dtypes(include=[np.number])
        X = X.fillna(0)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Labels distribution: {np.bincount(y.astype(int))}")
        
        return X.values, y
    
    def fit_scaler(self, X):
        """Fit standard scaler on training data"""
        self.scaler.fit(X)
        logger.info("✓ Scaler fitted")
    
    def transform(self, X):
        """Transform data using fitted scaler"""
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.scaler.fit_transform(X)
