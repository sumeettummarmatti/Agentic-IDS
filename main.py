import os
import argparse
import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Load environment variables FIRST
load_dotenv(dotenv_path='config/.env')

from src.detector.ensemble_model import EnsembleDetector
from src.detector.preprocessor import Preprocessor
from src.agents.attacker_agents import generate_balanced_synthetic_dataset
from src.agents.defender_agent import DefenderRLAgent
from src.council.llm_council_wrapper import ThreatAnalysisCouncil

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Agentic IDS')
    parser.add_argument('--live-data', type=str, help='Path to custom Excel file for live monitoring simulation')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*80)
    print("STARTING AGENTIC IDS PIPELINE")
    if args.live_data:
        print(f"MODE: Custom Live Data ({args.live_data})")
    else:
        print("MODE: Standard Simulation")
    print("="*80)

    # --- 1. INITIALIZE AGENTS ---
    detector = EnsembleDetector(use_lstm=True)
    council = ThreatAnalysisCouncil()
    defender = DefenderRLAgent()
    preprocessor = Preprocessor()
    
    # --- 2. DATA LOADING & TRAINING ---
    logger.info("\n[PHASE 1] Data & Training (Simulation)")
    
    user_data_path = 'data/raw/filtered_nowebatt.csv'
    default_data_path = 'data/raw/Darknet.xlsx'
    
    if os.path.exists(user_data_path):
        logger.info(f"Loading training data from {user_data_path} (User Optimized)...")
        real_df = preprocessor.load_data(user_data_path)
        X_real, _, y_multi_real = preprocessor.prepare_features_and_labels(real_df, training=True)
        # Using user data means we don't need synthetic alignment for the demo
        X = X_real
        y = y_multi_real
        logger.info(f"Training on {len(X)} rows from {user_data_path}")
        
    else:
        logger.info(f"Loading training data from {default_data_path} (Simulation Default)...")
        real_df = preprocessor.load_data(default_data_path)
        X_real, _, y_multi_real = preprocessor.prepare_features_and_labels(real_df, training=True)
        
        # Synthetic generation for simulation mode
        synthetic_df = generate_balanced_synthetic_dataset(num_ddos=500, num_portscan=200)
        # Align features
        required_columns = preprocessor.feature_names
        for col in required_columns:
            if col not in synthetic_df.columns:
                synthetic_df[col] = 0
        X_syn = synthetic_df[required_columns].values
        # Encode labels if possible, else zeros
        y_multi_syn = np.zeros(len(synthetic_df)) 
        
        X = np.vstack([X_real, X_syn])
        y = np.hstack([y_multi_real, y_multi_syn])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # Scale (Already handled by Preprocessor)
    X_train_scaled = X_train
    X_test_scaled = X_test
    
    # Update detector references
    detector.scaler = preprocessor.scaler
    
    # Train Detector
    detector.train(X_train_scaled, y_train)
    defender.train(total_timesteps=500) 
    
    # --- 3. LIVE MONITORING LOOP ---
    logger.info("\n[PHASE 2] Live Monitoring Loop")
    
    if args.live_data:
        # Load and process the live file
        logger.info(f"Loading custom live data from: {args.live_data}")
        try:
            live_df = preprocessor.load_data(args.live_data)
            X_live, _, _ = preprocessor.prepare_features_and_labels(live_df, training=False)
            target_flows = X_live
            # Limit for demo purposes if too large
            if len(target_flows) > 5:
                logger.info("Limiting to first 5 flows for demo...")
                target_flows = target_flows[:5]
            indices = range(len(target_flows))
        except Exception as e:
            logger.error(f"Failed to load custom data: {e}")
            return
    else:
        logger.info("Simulating incoming flows (using Test Set)...")
        attack_indices = np.where(y_test > 0)[0]
        if len(attack_indices) > 0:
            indices = attack_indices[:3] 
            target_flows = X_test
        else:
            indices = []
            target_flows = []

    for i, idx in enumerate(indices):
        if args.live_data:
             flow_features = target_flows[idx] # direct index
        else:
             flow_features = target_flows[i] # target_flows is already a slice X_test[sample_indices]
             # Wait, if target_flows = X_test[sample_indices], then it's a list/array of length 3.
             # indices is usually [index1, index2, index3].
             # So i goes 0,1,2.
             # flow_features = target_flows[i] is correct.
        
        # DIRECT MODEL PREDICTION
        
        # XGBoost
        xgb_prob = detector.xgb_model.predict_proba(flow_features.reshape(1, -1))[0]
        
        # LSTM
        import torch
        features_tensor = torch.FloatTensor(flow_features).unsqueeze(0).unsqueeze(0) # (1, 1, F)
        with torch.no_grad():
            lstm_out = detector.lstm_model(features_tensor)
            lstm_prob = torch.softmax(lstm_out, dim=1).numpy()[0]
            
        # Ensemble Soft Voting
        final_prob = (0.7 * xgb_prob) + (0.3 * lstm_prob)
        prediction_idx = np.argmax(final_prob)
        confidence = final_prob[prediction_idx]
        
        # Map Index to Name
        class_names = {0: 'BENIGN', 1: 'DDoS', 2: 'PortScan'}
        
        if hasattr(detector, 'training_classes_'):
             try:
                 mapped_idx = detector.training_classes_[prediction_idx]
                 attack_type = class_names.get(mapped_idx, str(mapped_idx))
             except:
                 attack_type = str(prediction_idx)
        else:
             attack_type = class_names.get(prediction_idx, str(prediction_idx))
        
        logger.info(f"\n>>> INCOMING FLOW {i+1} <<<")
        logger.info(f"Detector: {attack_type} ({confidence:.1%} conf)")
        
        # Create a dummy dict for Council/Logging
        cols = preprocessor.feature_names if preprocessor.feature_names else [f"F{k}" for k in range(len(flow_features))]
        flow_dict = dict(zip(cols, flow_features))
        
        prediction = {'attack_type': attack_type, 'confidence': confidence}
        
        if confidence > 0.6: 
             logger.info("-> High confidence threat! Summoning Council...")
             council_result = council.analyze_threat(flow_dict, prediction)
             
             # Defender
             # Create perception
             perception = {
                 'confidence': confidence, 
                 'threat_level': 'High', # Assume high if council summoned
                 'flow_rate': flow_dict.get('Flow Packets/s', 0)
             }
             
             observation = defender.observe(perception)
             action_result = defender.act(observation)
             
             logger.info(f"DEFENDER ACTION: {action_result['action']}")
             logger.info(f"Action Status: {action_result['status']}")
             logger.info("-> Mitigation applied. Monitoring effect...")
            
        else:
            logger.info("-> Benign/Low confidence. No action taken.")
            
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
