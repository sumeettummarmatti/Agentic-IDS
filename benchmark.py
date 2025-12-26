import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from src.detector.preprocessor import Preprocessor
from src.detector.ensemble_model import EnsembleDetector

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_benchmark():
    print("="*80)
    print("BENCHMARKING OPTIMIZED ARCHITECTURE")
    print("="*80)
    
    # 1. Initialize
    preprocessor = Preprocessor()
    detector = EnsembleDetector(use_lstm=True) # Testing ensemble
    
    # 2. Load Data
    data_path = 'data/raw/filtered_nowebatt.csv'
    logger.info(f"Loading {data_path}...")
    df = preprocessor.load_data(data_path)
    
    # 3. Preprocess (Fit)
    logger.info("Preprocessing and feature selection...")
    X, y, _ = preprocessor.prepare_features_and_labels(df, training=True)
    
    logger.info(f"Final Feature Shape: {X.shape}")
    logger.info(f"Classes: {np.unique(y)}")
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Train
    logger.info("Training Ensemble...")
    detector.train(X_train, y_train)
    
    # 6. Evaluate
    logger.info("Evaluating on Test Set...")
    
    # Predict row by row (as the detector expects) or adapt detector to batch?
    # Detector.predict expects a dict or single row usually, but we can access internal models for batch 
    # OR we can just loop (slow but uses exact pipeline)
    # Actually, let's use the internal XGBoost model for a direct batch comparison first 
    # to see if the core model is good.
    
    y_pred = detector.xgb_model.predict(X_test)
    
    # Map back if needed (the detector handles mapping internally during training, 
    # so y_pred from xgb might need remapping if labels weren't 0,1,2 directly)
    # In preprocessor we mapped BENIGN:0, DDoS:1, PortScan:2
    # So they are contiguous 0,1,2. No remapping needed hopefully.
    
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*50)
    print(f"Test Accuracy (XGBoost Component): {acc:.4f}")
    print("="*50)
    
    print("\nClassification Report:")
    target_names = ['BENIGN', 'DDoS', 'PortScan'] # Assumed order 0,1,2
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

if __name__ == "__main__":
    run_benchmark()
