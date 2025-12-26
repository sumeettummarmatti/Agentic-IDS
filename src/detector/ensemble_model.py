"""
Ensemble Detector: XGBoost + LSTM for intrusion detection
Loads your trained models and provides real-time predictions
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib

logger = logging.getLogger(__name__)

class EnsembleDetector:
    """
    Multi-model detector combining XGBoost and LSTM
    
    Features:
    - Loads pre-trained models from disk
    - Handles feature preprocessing
    - Returns confidence scores + label
    - Tracks metrics for explainability
    """
    #checkonce - initialize the model with pkl file and other things 
    def __init__(self, 
                 xgboost_model_path='models/trained/xgboost_model.pkl',   
                 feature_scaler_path='models/trained/scaler.pkl',
                 feature_list_path='models/trained/feature_list.pkl'):
        """
        Initialize detector with pre-trained models
        
        Args:
            xgboost_model_path: Path to trained XGBoost model
            feature_scaler_path: Path to feature scaler
            feature_list_path: List of expected features
        """
        self.model_path = Path(xgboost_model_path)
        self.scaler_path = Path(feature_scaler_path)
        self.feature_list_path = Path(feature_list_path)
        
        self.model = None
        self.scaler = None
        self.feature_list = None
        self.prediction_history = []
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"✓ Loaded XGBoost model from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}. "
                             "Will train on first call.")
                self.model = None
                
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✓ Loaded scaler from {self.scaler_path}")
            else:
                self.scaler = StandardScaler()
                
            if self.feature_list_path.exists():
                self.feature_list = joblib.load(self.feature_list_path)
                logger.info(f"✓ Loaded {len(self.feature_list)} features")
            else:
                self.feature_list = None
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def preprocess_features(self, flow_dict):
        """
        Preprocess flow for model input
        
        Args:
            flow_dict: Network flow as dictionary
            
        Returns:
            np.array: Preprocessed feature vector
        """
        try:
            # Convert dict to DataFrame
            if isinstance(flow_dict, dict):
                df = pd.DataFrame([flow_dict])
            else:
                df = flow_dict.copy()
            
            # Select only numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Fill NaN with 0
            numeric_df = numeric_df.fillna(0)
            
            # Scale features
            if self.scaler:
                scaled = self.scaler.transform(numeric_df)
            else:
                scaled = numeric_df.values
            
            return scaled if scaled.ndim > 1 else scaled
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None
    
    def predict(self, flow_dict, return_proba=True):
        """
        Predict if flow is attack or benign
        
        Args:
            flow_dict: Network flow
            return_proba: Return confidence scores
            
        Returns:
            dict: {
                'prediction': 0/1 (benign/attack),
                'confidence': 0-1 (confidence score),
                'probability': [P(benign), P(attack)],
                'threat_level': 'Low/Medium/High',
                'timestamp': prediction_time
            }
        """
        from datetime import datetime
        
        try:
            # Preprocess
            features = self.preprocess_features(flow_dict)
            if features is None:
                return {
                    'prediction': None,
                    'confidence': 0.0,
                    'error': 'Preprocessing failed'
                }
            
            # Reshape for XGBoost
            X = features.reshape(1, -1)
            
            # Predict
            if self.model:
                prediction = self.model.predict(X)
                
                if return_proba:
                    proba = self.model.predict_proba(X)
                else:
                    proba = [1.0 - prediction, prediction]
                
                confidence = max(proba)
                threat_level = self._get_threat_level(confidence)
                
                result = {
                    'prediction': int(prediction),  # 0=benign, 1=attack
                    'confidence': float(confidence),
                    'probability': [float(proba), float(proba)],
                    'threat_level': threat_level,
                    'timestamp': datetime.now().isoformat(),
                    'features_used': len(features)
                }
            else:
                # Model not loaded
                result = {
                    'prediction': None,
                    'confidence': 0.0,
                    'error': 'Model not loaded'
                }
            
            # Track for metrics
            self.prediction_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'prediction': None, 'confidence': 0.0, 'error': str(e)}
    
    def predict_batch(self, flows_df):
        """
        Predict multiple flows at once
        
        Args:
            flows_df: DataFrame of flows
            
        Returns:
            list: Prediction results for each flow
        """
        results = []
        for idx, row in flows_df.iterrows():
            result = self.predict(row.to_dict())
            results.append(result)
        return results
    
    def _get_threat_level(self, confidence):
        """Convert confidence to threat level"""
        if confidence < 0.5:
            return 'Low'
        elif confidence < 0.75:
            return 'Medium'
        else:
            return 'High'
    
    def get_metrics(self):
        """Return detection metrics from history"""
        if not self.prediction_history:
            return {}
        
        history = self.prediction_history
        attacks_detected = sum(1 for p in history if p['prediction'] == 1)
        high_confidence = sum(1 for p in history 
                            if p['confidence'] > 0.9)
        
        return {
            'total_predictions': len(history),
            'attacks_detected': attacks_detected,
            'attack_rate': attacks_detected / len(history),
            'high_confidence_predictions': high_confidence,
            'avg_confidence': np.mean([p['confidence'] 
                                      for p in history])
        }
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model on your data
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        try:
            logger.info("Training XGBoost model...")
            
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
            
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=10 if eval_set else False
            )
            
            # Save model
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            logger.info(f"✓ Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
