"""
Complete Agentic IDS Pipeline
- Loads your data
- Trains detector
- Generates synthetic data
- Runs LLM council analysis
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Your modules
from src.detector.ensemble_model import EnsembleDetector
from src.detector.preprocessor import Preprocessor
from src.agents.attacker_agents import generate_balanced_synthetic_dataset
from src.council.llm_council_wrapper import ThreatAnalysisCouncil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Complete pipeline execution"""
    
    logger.info("=" * 70)
    logger.info("AGENTIC-IDS COMPLETE PIPELINE")
    logger.info("=" * 70)
    
    # Step 1: Load real data
    logger.info("\n[STEP 1] Loading CIC-Darknet2020 dataset...")
    preprocessor = Preprocessor()
    
    real_df = preprocessor.load_data('data/raw/Darknet.xlsx')
    X_real, y_real = preprocessor.prepare_features_and_labels(real_df)
    X_real_scaled = preprocessor.fit_transform(X_real)
    
    logger.info(f"✓ Loaded {len(X_real)} real flows")
    logger.info(f"  Features: {X_real.shape}")
    logger.info(f"  Class distribution: {np.bincount(y_real.astype(int))}")
    
    # Step 2: Generate synthetic attacks
    logger.info("\n[STEP 2] Generating synthetic attack data...")
    synthetic_df = generate_balanced_synthetic_dataset(
        num_ddos=2000,
        num_portscan=1000
    )
    synthetic_df.to_csv('data/synthetic/attacks.csv', index=False)
    logger.info(f"✓ Synthetic data saved to data/synthetic/attacks.csv")
    
    # Prepare synthetic data
    X_synthetic, y_synthetic = preprocessor.prepare_features_and_labels(synthetic_df)
    X_synthetic_scaled = preprocessor.transform(X_synthetic)
    
    # Step 3: Combine datasets
    logger.info("\n[STEP 3] Combining real + synthetic data...")
    X_combined = np.vstack([X_real_scaled, X_synthetic_scaled])
    y_combined = np.hstack([y_real, y_synthetic])
    
    logger.info(f"✓ Combined dataset: {len(X_combined)} flows")
    logger.info(f"  Benign: {(y_combined == 0).sum()}")
    logger.info(f"  Attack: {(y_combined == 1).sum()}")
    
    # Step 4: Train detector
    logger.info("\n[STEP 4] Training ensemble detector...")
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
    logger.info("\nDetection Performance:")
    logger.info(classification_report(y_test, y_pred))
    
    # Step 5: Initialize threat analysis council
    logger.info("\n[STEP 5] Initializing LLM Council...")
    council = ThreatAnalysisCouncil(
        enable_groq=True,  # Uses free Groq API
        enable_ollama=True  # Falls back to local LLM
    )
    
    # Step 6: Analyze sample attacks
    logger.info("\n[STEP 6] Running council analysis on detected attacks...")
    
    # Find attacks in test set
    attack_indices = np.where(y_test == 1)[:5]
    
    for idx, test_idx in enumerate(attack_indices):
        logger.info(f"\n--- Analyzing Attack {idx + 1} ---")
        
        # Get flow data
        flow_features = X_test[test_idx]
        flow_dict = dict(zip(
            preprocessor.feature_names,
            flow_features
        ))
        
        # Get detector prediction
        detector_pred = detector.predict(flow_dict)
        
        # Get council analysis
        analysis = council.analyze_threat(flow_dict, detector_pred)
        
        # Log results
        logger.info(f"Threat Type: {analysis.threat_type}")
        logger.info(f"Severity: {analysis.severity}")
        logger.info(f"Council Consensus: {analysis.council_consensus:.2%}")
        logger.info(f"Explanation:\n{analysis.explanation[:300]}...")
        logger.info(f"Recommendations: {analysis.recommendations}")
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nDetector Metrics:")
    logger.info(detector.get_metrics())

if __name__ == "__main__":
    main()
