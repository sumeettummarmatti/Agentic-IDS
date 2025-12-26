"""
Agentic IDS - Full Pipeline
"""

import pandas as pd
import numpy as np
import logging
import time
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Load environment variables FIRST
load_dotenv(dotenv_path='config/.env')

import argparse
from src.detector.ensemble_model import EnsembleDetector
from src.detector.preprocessor import Preprocessor
from src.agents.attacker_agents import generate_balanced_synthetic_dataset
from src.council.llm_council_wrapper import ThreatAnalysisCouncil
from src.agents.defender_agent import DefenderRLAgent

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
    # Detector (Ensemble: XGBoost + LSTM)
    detector = EnsembleDetector(use_lstm=True)
    
    # Council (LLM Multi-Agent System)
    council = ThreatAnalysisCouncil()
    
    # Defender (RL Agent)
    defender = DefenderRLAgent()
    
    # Preprocessor
    preprocessor = Preprocessor()
    
    # --- 2. DATA LOADING & TRAINING ---
    logger.info("\n[PHASE 1] Data & Training (Simulation)")
    
    # Load Real Data (Darknet) for TRAINING
    logger.info("Loading training data from data/raw/Darknet.xlsx...")
    real_df = preprocessor.load_data('data/raw/Darknet.xlsx') # Always train on base data
    synthetic_df = generate_balanced_synthetic_dataset(num_ddos=500, num_portscan=200)
    
    # Prepare Labels
    # Map synthetic labels to mimic the dataset classes for consistency
    synthetic_df['Label'] = synthetic_df['Label'].map({
        'DDoS': 'P2P', 'PortScan': 'FILE-TRANSFER' 
    }) # Mapping to closest existing classes for this demo
    
    X_real, _, y_multi_real = preprocessor.prepare_features_and_labels(real_df)
    
    # Align synthetic data features with real data
    required_columns = preprocessor.feature_names
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in synthetic_df.columns:
            synthetic_df[col] = 0
            
    # Extract X_syn using PRECISELY the same columns
    X_syn = synthetic_df[required_columns].values
    
    # Extract labels manually since we might have dropped columns preprocessor needs for y or to avoid re-fit
    # But we can use the preprocessor helper if we are careful, but doing it manually is safer here
    if 'Label' in synthetic_df.columns:
        # Filter unknown (already should be clean from generator but good to be safe)
        synthetic_df = synthetic_df[synthetic_df['Label'].isin(preprocessor.ATTACK_CLASSES)]
        # Use the SAME encoder instance
        y_multi_syn = preprocessor.label_encoder.transform(synthetic_df['Label'])
    else:
        y_multi_syn = np.zeros(len(synthetic_df)) # Should not happen
        
    # We also need to filter X_syn if we filtered df due to labels, so let's do it in order
    X_syn = synthetic_df[required_columns].values
    
    logger.info(f"Real features: {X_real.shape}, Syn features: {X_syn.shape}")
    
    # Check
    if X_real.shape[1] != X_syn.shape[1]:
        raise ValueError(f"Feature mismatch after fix! {X_real.shape[1]} vs {X_syn.shape[1]}")
    
    # Combine
    X = np.vstack([X_real, X_syn])
    y = np.hstack([y_multi_real, y_multi_syn])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # Scale
    preprocessor.fit_scaler(X_train)
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Update detector references
    detector.scaler = preprocessor.scaler
    
    # Train Detector
    detector.train(X_train_scaled, y_train)
    defender.train(total_timesteps=500) # Pre-train defender
    
    # --- 3. LIVE MONITORING LOOP ---
    logger.info("\n[PHASE 2] Live Monitoring Loop")
    
    if args.live_data:
        logger.info(f"Loading custom live data from: {args.live_data}")
        try:
            live_df = pd.read_excel(args.live_data)
            logger.info(f"Loaded {len(live_df)} flows from custom file.")
            
            # Preprocess using the SAME preprocessor fitted on training data
            # Note: We need to handle potential missing columns by aligning features
            # This logic mimics what we did for synthetic data earlier
            
            # 1. Align columns
            required_cols = preprocessor.feature_names
            for col in required_cols:
                if col not in live_df.columns:
                    live_df[col] = 0
            
            # 2. Extract features strictly
            # We don't have labels usually in live data, so we just get X
            # But the preprocessor.prepare might expect labels.
            # Let's extract X manually using the scaler
            X_live_raw = live_df[required_cols].values
            X_live_raw = np.nan_to_num(X_live_raw)
            
            # 3. Scale
            if preprocessor.scaler:
                 X_live_scaled = preprocessor.scaler.transform(X_live_raw)
            else:
                 X_live_scaled = X_live_raw
                 
            # 4. Iterate
            target_flows = X_live_scaled
            logger.info(f"Processing {len(target_flows)} custom flows...")
            
        except Exception as e:
            logger.error(f"Failed to load/process custom data: {e}")
            return
            
    else:
        logger.info("Simulating incoming flows (using Test Set)...")
        # Pick some interesting test cases (Attacks)
        attack_indices = np.where(y_test > 0)[0]
        if len(attack_indices) > 0:
            sample_indices = attack_indices[:3] # Analyze 3 attacks
            target_flows = X_test_scaled[sample_indices]
        else:
            logger.warning("No attacks found in test set to simulate.")
            target_flows = []
    
    for idx, flow_features_scaled in enumerate(target_flows):
        # We need to map back to a dict for the 'predict' method which might expect raw values 
        # OR the predict method expects scaled values if we pass a dict?
        # Checking ensemble_model.py: predict() calls _preprocess_features() which cleans and SCALES.
        # WAIT! If 'predict' SCALES internally, we should pass RAW properties if we pass a dict.
        # But if we pass a numpy array to the internal models, they expect scaled.
        # The 'predict' method takes 'flow_dict'.
        
        # Let's look at how we constructed 'flow_dict' before.
        # "flow_features = X_test[test_idx]" -> This was RAW X_test (before scaling) in previous code?
        # No, in previous code:
        # X_train, X_test... then preprocessor.fit_scaler(X_train) -> X_train_scaled.
        # In previous code "flow_features = X_test[test_idx]" used X_test which was UNSCALED.
        
        # Checking logic above:
        # X_train, X_test = train_test_split(X...) -> Raw
        # X_train_scaled = transform(X_train)
        
        # So 'target_flows' should be RAW data because 'detector.predict' will scale it.
        pass
        
    # CORRECTION: We need RAW data for the loop because detector.predict(flow_dict) applies preprocessing/scaling internally.
    
    if args.live_data:
        # We already loaded X_live_raw
        target_indices = range(len(X_live_raw))
        source_data = X_live_raw
    else:
        # Use X_test (RAW)
        attack_indices = np.where(y_test > 0)[0]
        if len(attack_indices) > 3:
             target_indices = attack_indices[:3]
        else:
             target_indices = attack_indices
        source_data = X_test

    for i, idx_ptr in enumerate(target_indices):
        if args.live_data:
             flow_features = source_data[idx_ptr] # it is just the index in X_live_raw
        else:
             flow_features = source_data[idx_ptr]
             
        # Create a dummy dict key-value for the detector
        if not preprocessor.feature_names: 
            cols = [f"Feature_{i}" for i in range(len(flow_features))]
        else:
            cols = preprocessor.feature_names
            
        flow_dict = dict(zip(cols, flow_features))
        
        logger.info(f"\n>>> INCOMING FLOW {i+1} <<<")
        
        # A. DETECTOR
        prediction = detector.predict(flow_dict)
        logger.info(f"Detector: {prediction['attack_type']} ({prediction['confidence']:.1%} conf)")
        
        if prediction['confidence'] > 0.6:
            # B. COUNCIL
            logger.info("-> High confidence threat! Summoning Council...")
            
            # Recover some raw values for the prompt (formatted nicely)
            analysis_data = {
                'Protocol': float(flow_features[0]), # Assumption on index
                'Total Fwd Packet': float(flow_features[1]),
                'Total Bwd packets': float(flow_features[2]),
                'Fwd SYN Flags': float(flow_features[preprocessor.feature_names.index('Fwd SYN Flags')]) if 'Fwd SYN Flags' in preprocessor.feature_names else 0,
                'Flow Packets/s': float(flow_features[preprocessor.feature_names.index('Flow Packets/s')]) if 'Flow Packets/s' in preprocessor.feature_names else 0
            }
            
            council_result = council.analyze_threat(analysis_data, prediction)
            
            # C. DEFENDER
            logger.info("-> Council Consensus Reached. Activating Defender...")
            
            # Create perception for defender
            perception = {
                'confidence': council_result.confidence,
                'threat_level': council_result.severity,
                'flow_rate': analysis_data['Flow Packets/s']
            }
            
            observation = defender.observe(perception)
            action = defender.act(observation)
            
            logger.info(f"DEFENDER ACTION: {action['action']}")
            logger.info(f"Action Status: {action['status']}")
            
            # Feedback (Dummy)
            logger.info("-> Mitigation applied. Monitoring effect...")
            
        else:
            logger.info("-> Benign/Low confidence. No action taken.")
            
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
