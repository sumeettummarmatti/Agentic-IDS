import logging
import numpy as np
import pandas as pd
from src.detector.ensemble_model import EnsembleDetector
from src.detector.preprocessor import Preprocessor
from src.agents.attacker_agents import DDoSAgent, PortScanAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("StressTest")

def run_stress_test():
    print("="*80)
    print("AGENTIC IDS: ADVERSARIAL STRESS TEST")
    print("="*80)
    
    # 1. Load Preprocessor (to get feature names and scaler)
    # We assume the preprocessor was saved or we init a new one and try to match logic
    # In a real app we would pickle the preprocessor. Here we assume the feature list matches.
    preprocessor = Preprocessor()
    
    # 2. Init Agents
    print("\n[INIT] Initializing Attacker Agents...")
    # High evasion level = more randomness/noise to confuse model
    ddos_agent = DDoSAgent(attack_type='syn_flood', evasion_level=1.0) 
    portscan_agent = PortScanAgent(scan_speed='fast') # Detection is harder if mixed with noise?
    
    # 3. Generate Adversarial Data
    print("[ATTACK] Generating Synthetic Adversarial Flows...")
    
    # Generate 10 DDoS flows with high evasion
    ddos_flows = []
    for _ in range(10):
        # Omega=2.0 means 2x intensity, evasion=True adds noise
        flow = ddos_agent.generate_flow(omega=2.0, use_evasion=True)
        ddos_flows.append(flow)
    ddos_df = pd.DataFrame(ddos_flows)
    
    # Generate 10 PortScan flows
    portscan_flows = []
    for _ in range(10):
        # Decoy ratio 0.5 means 50% of traffic is fake "normal" traffic to hide the scan
        flow = portscan_agent.generate_flow(decoy_ratio=0.5)
        portscan_flows.append(flow)
    portscan_df = pd.DataFrame(portscan_flows)
    
    # Combine
    attack_df = pd.concat([ddos_df, portscan_df], ignore_index=True)
    print(f"✓ Generated {len(attack_df)} adversarial flows")
    
    # 4. Preprocess for Model
    print("[PREPROCESS] aligning features with trained model...")
    # Load feature names expected by the trained model (hardcoded from main.py logic for now or we load from file)
    # Ideally, we load the list from the saved model metadata, but we'll infer it via Preprocessor logic
    # We need to ensure columns match exactly what the model expects (50 features)
    
    # Hack: We use the same alignment logic as main.py
    # Load a tiny dummy real file to get feature names if needed, OR just trust the agent output
    # The agents produce 'Protocol', 'Total Fwd Packet', etc.
    # The Preprocessor.feature_selection dropped many.
    
    # 4. Preprocess for Model
    print("[PREPROCESS] aligning features with trained model...")
    
    try:
        # Load reference data to fit the pipeline exactly as main.py did
        real_df = preprocessor.load_data('data/raw/filtered_nowebatt.csv')
        # Fit pipeline
        _ = preprocessor.prepare_features_and_labels(real_df, training=True)
        print(f"✓ Pipeline fitted on {len(real_df)} rows")
        
        # Align attack data to match the RAW input schema (80 cols)
        # Note: We must exclude 'Label' from features, as Preprocessor drops it during fit (if string)
        # but if we add it as 0 (int) here, it gets included as a feature -> Mismatch.
        
        required_raw_cols = [c for c in real_df.columns if c != 'Label']
        
        # 1. Add missing columns
        for col in required_raw_cols:
            if col not in attack_df.columns:
                attack_df[col] = 0 
                
        # 2. Filter and Reorder to match exactly
        attack_df = attack_df[required_raw_cols]
        
        # Now use the pipeline to transform (Clean -> Scale -> Select)
        # This returns the exact 50 features expected by the model
        if hasattr(preprocessor.scaler, "feature_names_in_"):
             # Sklearn 1.0+ validates feature names. We must match exactly.
             pass
             
        X_attack_scaled, _, _ = preprocessor.prepare_features_and_labels(attack_df, training=False)
        print(f"✓ Attack data processed to shape: {X_attack_scaled.shape}")
        
    except Exception as e:
        print(f"Error referencing data: {e}")
        return

    # 5. Load & Test Model
    print("[DEFENSE] Testing Ensemble Detector against attacks...")
    detector = EnsembleDetector(use_lstm=True)
    # Force load the saved models
    try:
        import joblib
        detector.xgb_model = joblib.load('models/trained/xgboost_multiclass.pkl')
        print("✓ XGBoost model loaded")
        
        # Load mapping if checks against classes needed, but we used implicit mapping in main
        # In main.py: {0: 'BENIGN', 1: 'DDoS', 2: 'PortScan'} based on user logic
    except Exception as e:
        print(f"XGBoost model not found or failed to load! Run main.py first. Error: {e}")
        return

    # Predict
    # Note: LSTM might fail if dimensions strictly mismatch or if we didn't save it properly to reload
    # For this stress test, we will check XGBoost primarily as it's the main classifier
    
    xgb_preds_proba = detector.xgb_model.predict_proba(X_attack_scaled)
    xgb_preds = np.argmax(xgb_preds_proba, axis=1)
    
    # 6. Analyze Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"{'Attack Type':<20} | {'Prediction':<20} | {'Confidence':<10} | {'Status'}")
    print("-" * 65)
    
    detected_count = 0
    for i, pred_class in enumerate(xgb_preds):
        # True Type
        true_type = "DDoS" if i < 10 else "PortScan" # First 10 are DDoS, next 10 PortScan
        
        # Prediction Mapping (0=Benign, 1=DDoS, 2=PortScan)
        # Note: This mapping depends on what main.py used.
        # main.py used: {'BENIGN': 0, 'DDoS': 1, 'PortScan': 2}
        
        pred_label = "BENIGN"
        if pred_class == 1: pred_label = "DDoS"
        if pred_class == 2: pred_label = "PortScan"
        
        conf = np.max(xgb_preds_proba[i])
        
        # Status
        status = "✅ DETECTED"
        if pred_label == "BENIGN":
            status = "❌ EVADED"
        elif pred_label != true_type:
            status = "⚠️ MISCLASSIFIED" # e.g. DDoS detected as PortScan (still blocked, but wrong type)
        else:
            detected_count += 1
            
        print(f"{true_type:<20} | {pred_label:<20} | {conf:.1%}    | {status}")

    accuracy = detected_count / len(X_attack_scaled)
    print("-" * 65)
    print(f"Stress Test Detection Rate: {accuracy:.1%}")
    print("="*80)

if __name__ == "__main__":
    run_stress_test()
