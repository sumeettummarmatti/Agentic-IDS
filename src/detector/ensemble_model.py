"""
Ensemble Detector: XGBoost for MULTI-LABEL classification
Detects specific attack types, not just benign/attack
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

class EnsembleDetector:
    """
    Multi-label detector using XGBoost
    
    Predicts specific attack types:
    - AUDIO-STREAMING, VIDEO-STREAMING, VOIP, etc.
    """
    
    def __init__(self, 
                 model_path='models/trained/xgboost_multiclass.pkl',
                 scaler_path='models/trained/scaler.pkl',
                 feature_list_path='models/trained/feature_list.pkl'):
        """Initialize detector"""
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.feature_list_path = Path(feature_list_path)
        
        self.model = None
        self.scaler = None
        self.feature_list = None
        self.class_names = None
        self.prediction_history = []
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"✓ Loaded XGBoost model")
            
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✓ Loaded scaler")
            
            if self.feature_list_path.exists():
                self.feature_list = joblib.load(self.feature_list_path)
                logger.info(f"✓ Loaded {len(self.feature_list)} features")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict(self, flow_dict, return_proba=True):
        """
        Predict attack type for a flow
        
        Returns:
            dict: {
                'attack_type': 'VOIP',  # Specific type
                'confidence': 0.92,
                'probabilities': {...},  # All class probabilities
                'threat_level': 'Medium'
            }
        """
        from datetime import datetime
        
        try:
            # Preprocess
            features = self._preprocess_features(flow_dict)
            if features is None:
                return {'error': 'Preprocessing failed'}
            
            # Predict
            if self.model:
                prediction = self.model.predict(features.reshape(1, -1))
                
                if return_proba:
                    proba = self.model.predict_proba(features.reshape(1, -1))
                else:
                    proba = np.zeros(self.model.n_classes_)
                    proba[prediction] = 1.0
                
                confidence = max(proba)
                
                # Map to class names
                attack_type = self._get_class_name(prediction)
                
                # Determine threat level based on attack type
                threat_level = self._get_threat_level(attack_type, confidence)
                
                result = {
                    'attack_type': attack_type,  # ← Specific type!
                    'confidence': float(confidence),
                    'probabilities': {
                        self._get_class_name(i): float(p) 
                        for i, p in enumerate(proba)
                    },
                    'threat_level': threat_level,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                result = {'error': 'Model not loaded'}
            
            self.prediction_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}
    
    def _preprocess_features(self, flow_dict):
        """Extract and scale features"""
        try:
            if isinstance(flow_dict, dict):
                df = pd.DataFrame([flow_dict])
            else:
                df = flow_dict.copy()
            
            numeric_df = df.select_dtypes(include=[np.number])
            numeric_df = numeric_df.fillna(0)
            
            if self.scaler:
                scaled = self.scaler.transform(numeric_df)
            else:
                scaled = numeric_df.values
            
            return scaled if scaled.ndim > 1 else scaled
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None
    
    def _get_class_name(self, class_idx):
        """Map class index to attack type name"""
        class_names = [
            'AUDIO-STREAMING', 'VIDEO-STREAMING', 'VOIP', 
            'BROWSING', 'FILE-TRANSFER', 'P2P', 'EMAIL', 
            'CHAT', 'STREAMING', 'TORRENT'
        ]
        return class_names[class_idx] if class_idx < len(class_names) else 'Unknown'
    
    def _get_threat_level(self, attack_type, confidence):
        """
        Determine threat severity based on attack type
        """
        # High-threat attacks
        high_threat = ['P2P', 'TORRENT', 'FILE-TRANSFER']
        medium_threat = ['VIDEO-STREAMING', 'AUDIO-STREAMING', 'STREAMING']
        low_threat = ['BROWSING', 'EMAIL', 'CHAT', 'VOIP']
        
        if attack_type in high_threat:
            return 'High' if confidence > 0.7 else 'Medium'
        elif attack_type in medium_threat:
            return 'Medium' if confidence > 0.8 else 'Low'
        else:
            return 'Low'
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost on multi-class labels
        
        Args:
            y_train: Multi-class labels (0-9 for 10 attack types)
        """
        try:
            logger.info("Training XGBoost for MULTI-CLASS classification...")
            
            eval_set = [(X_val, y_val)] if X_val is not None else None
            
            # Use XGBClassifier with num_class for multi-class
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                num_class=10,  # 10 attack types
                objective='multi:softmax',
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=10 if eval_set else False
            )
            
            # Save
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            logger.info(f"✓ Multi-class model saved")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
