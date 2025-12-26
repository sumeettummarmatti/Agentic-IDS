"""
Multi-label IDS pipeline
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.detector.ensemble_model import EnsembleDetector
from src.detector.preprocessor import Preprocessor
from src.agents.attacker_agents import generate_balanced_synthetic_dataset
from src.council.llm_council_wrapper import ThreatAnalysisCouncil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Multi-label classification pipeline"""
    
    logger.info("=" * 70)
    logger.info("AGENTIC-IDS: MULTI-LABEL ATTACK CLASSIFICATION")
    logger.info("=" * 70)
    
    # Step 1: Load real data
    logger.info("\n[STEP 1] Loading CIC-Darknet2020 dataset...")
    preprocessor = Preprocessor(use_multi_label=True)
    
    real_df = preprocessor.load_data('data/raw/Darknet.xlsx')
    X_real, y_binary, y_multi = preprocessor.prepare_features_and_labels(real_df)
    X_real_scaled = preprocessor.fit_transform(X_real)
    
    logger.info(f"✓ Loaded {len(X_real)} real flows")
    
    # Step 2: Generate synthetic attacks
    logger.info("\n[STEP 2] Generating synthetic attack data...")
    synthetic_df = generate_balanced_synthetic_dataset(
        num_ddos=2000,
        num_portscan=1000
    )
    
    # Label synthetic data
    synthetic_df['Label'] = synthetic_df['Label'].map({
        'DDoS': 'P2P',           # Map to attack type
        'PortScan': 'FILE-TRANSFER'
    })
    
    X_synthetic, _, y_synthetic_multi = preprocessor.prepare_features_and_labels(synthetic_df)
    X_synthetic_scaled = preprocessor.transform(X_synthetic)
    
    # Step 3: Combine and train
    logger.info("\n[STEP 3] Training multi-class detector...")
    X_combined = np.vstack([X_real_scaled, X_synthetic_scaled])
    y_combined = np.hstack([y_multi, y_synthetic_multi])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined,
        test_size=0.2,
        random_state=42,
        stratify=y_combined
    )
    
    detector = EnsembleDetector()
    detector.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    y_pred = detector.model.predict(X_test)
    logger.info("\nMulti-Class Classification Report:")
    logger.info(classification_report(y_test, y_pred, 
                target_names=[
                    'AUDIO', 'VIDEO', 'VOIP', 'BROWSE', 
                    'FILE', 'P2P', 'EMAIL', 'CHAT', 'STREAM', 'TORRENT'
                ]))
    
    # Step 4: Council analysis
    logger.info("\n[STEP 4] Running council analysis...")
    council = ThreatAnalysisCouncil()
    
    # Analyze sample flows
    attack_indices = np.where(y_test > 0)[:5]
    
    for idx, test_idx in enumerate(attack_indices):
        flow_features = X_test[test_idx]
        flow_dict = dict(zip(
            preprocessor.feature_names,
            flow_features
        ))
        
        prediction = detector.predict(flow_dict)
        
        logger.info(f"\n--- Attack {idx + 1} ---")
        logger.info(f"Type: {prediction['attack_type']}")
        logger.info(f"Confidence: {prediction['confidence']:.2%}")
        logger.info(f"Threat Level: {prediction['threat_level']}")

if __name__ == "__main__":
    main()
