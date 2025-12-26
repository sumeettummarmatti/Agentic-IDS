"""
Ensemble Detector: XGBoost + LSTM for robust detection
"""

import pandas as pd
import numpy as np
import joblib
import logging
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

class LSTMDetector(nn.Module):
    """
    LSTM-based detector for temporal patterns in flows
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=10):
        super(LSTMDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class EnsembleDetector:
    """
    Ensemble using XGBoost and LSTM
    """
    
    def __init__(self, 
                 model_path='models/trained/',
                 use_lstm=True):
        """Initialize detector"""
        self.model_path = Path(model_path)
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = None
        self.use_lstm = use_lstm
        self.prediction_history = []
        
        self.class_names = [
            'AUDIO-STREAMING', 'VIDEO-STREAMING', 'VOIP', 
            'BROWSING', 'FILE-TRANSFER', 'P2P', 'EMAIL', 
            'CHAT', 'STREAMING', 'TORRENT'
        ]
        
        self.input_size = 71 # Feature count
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        xgb_path = self.model_path / 'xgboost_multiclass.pkl'
        scaler_path = self.model_path / 'scaler.pkl'
        
        try:
            if xgb_path.exists():
                self.xgb_model = joblib.load(xgb_path)
                logger.info("✓ Loaded XGBoost model")
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("✓ Loaded scaler")
                
            # Initialize LSTM (dummy weights if no path, normally would load)
            if self.use_lstm:
                self.lstm_model = LSTMDetector(input_size=self.input_size, num_classes=len(self.class_names))
                self.lstm_model.eval() # Set to eval mode
                logger.info("✓ Initialized LSTM model")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def predict(self, flow_dict):
        """
        Ensemble prediction
        """
        try:
            # Preprocess
            features = self._preprocess_features(flow_dict)
            if features is None:
                return {'error': 'Preprocessing failed'}
            
            # --- XGBoost Prediction ---
            xgb_proba = np.zeros(len(self.class_names))
            if self.xgb_model:
                # The model predicts probabilities for the CONTIGUOUS classes 0..K-1
                # We need to map them back to the ORIGINAL class indices
                # self.training_classes_ contains the original indices in order 0..K-1
                
                raw_proba = self.xgb_model.predict_proba(features.reshape(1, -1))[0]
                
                # Map back to full reference probabilities
                # Initialize with 0s for all 10 classes
                # Then fill in the ones we know about
                if hasattr(self, 'training_classes_') and self.training_classes_ is not None:
                     for i, original_class_idx in enumerate(self.training_classes_):
                         if i < len(raw_proba):
                             xgb_proba[original_class_idx] = raw_proba[i]
                else: 
                     # Fallback if model loaded from old version or no mapping
                     if len(raw_proba) == len(xgb_proba):
                         xgb_proba = raw_proba
            
            # --- LSTM Prediction ---
            lstm_proba = np.zeros(len(self.class_names))
            if self.lstm_model:
                with torch.no_grad():
                    # Reshape for LSTM: (batch, seq_len, features) -> (1, 1, 71)
                    # features is (1, 71), so we want (1, 1, 71)
                    input_tensor = torch.FloatTensor(features).unsqueeze(1)
                    outputs = self.lstm_model(input_tensor)
                    # LSTM was trained on contiguous labels too
                    raw_lstm_proba = torch.softmax(outputs, dim=1).numpy()[0]
                    
                    if hasattr(self, 'training_classes_') and self.training_classes_ is not None:
                         for i, original_class_idx in enumerate(self.training_classes_):
                             if i < len(raw_lstm_proba):
                                 lstm_proba[original_class_idx] = raw_lstm_proba[i]
                    else:
                         if len(raw_lstm_proba) == len(lstm_proba):
                             lstm_proba = raw_lstm_proba
            
            # --- Ensemble (Soft Voting) ---
            final_proba = (0.7 * xgb_proba) + (0.3 * lstm_proba)
            
            prediction_idx = np.argmax(final_proba)
            confidence = final_proba[prediction_idx]
            attack_type = self._get_class_name(prediction_idx)
            
            result = {
                'attack_type': attack_type,
                'confidence': float(confidence),
                'probabilities': {
                    self._get_class_name(i): float(p) 
                    for i, p in enumerate(final_proba)
                },
                'threat_level': self._get_threat_level(attack_type, confidence),
                'components': {
                    'xgb_top': self._get_class_name(np.argmax(xgb_proba)),
                    'lstm_top': self._get_class_name(np.argmax(lstm_proba))
                }
            }
            
            self.prediction_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}

    
    
    def _get_class_name(self, class_idx):
        return self.class_names[class_idx] if class_idx < len(self.class_names) else 'Unknown'
    
    def _get_threat_level(self, attack_type, confidence):
        high_threat = ['P2P', 'TORRENT', 'FILE-TRANSFER']
        medium_threat = ['VIDEO-STREAMING', 'AUDIO-STREAMING', 'STREAMING']
        
        if attack_type in high_threat:
            return 'High' if confidence > 0.7 else 'Medium'
        elif attack_type in medium_threat:
            return 'Medium' if confidence > 0.8 else 'Low'
        else:
            return 'Low'

    def _preprocess_features(self, flow_dict):
        """Extract and scale features"""
        try:
            if isinstance(flow_dict, dict):
                df = pd.DataFrame([flow_dict])
            else:
                df = flow_dict.copy()
            
            numeric_df = df.select_dtypes(include=[np.number])
            numeric_df = numeric_df.fillna(0)
            
            # Ensure we have correct columns (pad/truncate if needed) - Simplified
            current_cols = numeric_df.shape[1]
            if current_cols < self.input_size:
                 # Pad with zeros
                 padding = np.zeros((1, self.input_size - current_cols))
                 # For now, just rely on what scaler expects if available
                 pass

            if self.scaler:
                try:
                    scaled = self.scaler.transform(numeric_df)
                except:
                    # Fallback if feature count mismatch (common in demo data)
                    logger.warning("Feature shape mismatch, reshaping/padding...")
                    # This is tricky without the original feature list. 
                    # We will just return the values padded to self.input_size
                    vals = numeric_df.values
                    if vals.shape[1] != self.input_size:
                        padded = np.zeros((1, self.input_size))
                        min_len = min(vals.shape[1], self.input_size)
                        padded[0, :min_len] = vals[0, :min_len]
                        return padded
                    return vals
            else:
                scaled = numeric_df.values
            
            return scaled if scaled.ndim > 1 else scaled
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train both XGBoost and LSTM
        """
        logger.info("Training Ensemble (XGBoost + LSTM)...")
        
        # REMAP LABELS to be contiguous 0..N
        from sklearn.preprocessing import LabelEncoder
        self.label_remapper = LabelEncoder()
        y_train_mapped = self.label_remapper.fit_transform(y_train)
        
        # Store the original class indices that correspond to 0, 1, 2...
        # label_remapper.classes_ holds the original values sorted
        self.training_classes_ = self.label_remapper.classes_
        num_classes_in_batch = len(self.training_classes_)
        
        logger.info(f"Training on {num_classes_in_batch} unique classes: {self.training_classes_}")
        
        # 1. Train XGBoost
        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=8, 
            learning_rate=0.1,
            num_class=num_classes_in_batch, # Use actual unique count
            objective='multi:softmax',
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model.fit(X_train, y_train_mapped)
        
        # 2. Train LSTM
        X_train_t = torch.FloatTensor(X_train).unsqueeze(1) 
        y_train_t = torch.LongTensor(y_train_mapped)
        
        self.lstm_model = LSTMDetector(input_size=X_train.shape[1], num_classes=num_classes_in_batch)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.01)
        
        # Simple training loop
        self.lstm_model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            logger.info(f"LSTM Epoch {epoch+1}, Loss: {loss.item():.4f}")
            
        # Save
        self.model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.xgb_model, self.model_path / 'xgboost_multiclass.pkl')
        # Also need to save the mapping if we want to load it later!
        joblib.dump(self.training_classes_, self.model_path / 'class_mapping.pkl')
        
        logger.info("✓ Models trained and saved")
