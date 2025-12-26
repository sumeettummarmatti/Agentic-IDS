#!/usr/bin/env python
"""
Main entry point: Agentic IDS Pipeline
1. Generate synthetic data via agents
2. Train detector
3. Deploy LLM council for explainability
"""

import os
import sys
import logging
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.attacker_agents import DDoSAgent, PortScanAgent, generate_balanced_synthetic_dataset
from src.detector.ensemble_model import EnsembleDetector
from src.detector.preprocessor import DataPreprocessor
from src.council.llm_council_wrapper import ThreatAnalysisCouncil
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Main pipeline"""
    
    logger.info("=" * 60)
    logger.info("AGENTIC IDS PIPELINE START")
    logger.info("=" * 60)
    
    # Step 1: Generate Synthetic Data
    logger.info("\n[Step 1] Generating synthetic attack data via agents...")
    synthetic_data = generate_balanced_synthetic_dataset(
        num_ddos=1000,  # Start small for testing
        num_portscan=500
    )
    synthetic_data.to_csv("data/synthetic/attacks.csv", index=False)
    logger.info(f"✓ Generated {len(synthetic_data)} synthetic flows")
    
    # Step 2: Load and prepare data
    logger.info("\n[Step 2] Loading and preprocessing data...")
    import pandas as pd
    real_data = pd.read_excel("data/raw/Darknet.xlsx")  # Your existing data
    
    # Combine
    combined = pd.concat([real_data, synthetic_data], ignore_index=True)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X = preprocessor.fit_transform(combined.select_dtypes(include=[np.number]))
    
    logger.info(f"✓ Prepared {X.shape} samples with {X.shape} features")
    
    # Step 3: Train detector
    logger.info("\n[Step 3] Training detector...")
    detector = EnsembleDetector()
    detector.train_xgboost(X, combined['Label'].values)
    logger.info("✓ Detector trained")
    
    # Step 4: Initialize council
    logger.info("\n[Step 4] Initializing LLM council...")
    council = ThreatAnalysisCouncil(use_local_llm=True, use_groq=True)
    logger.info("✓ Council ready")
    
    # Step 5: Test on sample
    logger.info("\n[Step 5] Testing on sample flow...")
    sample_flow = combined.iloc.to_dict()
    
    predictions = detector.predict(X[:1])
    analysis = council.analyze_threat(sample_flow)
    
    logger.info(f"Prediction: {predictions}")
    logger.info(f"Council Analysis: {analysis.attack_type.value} ({analysis.confidence}%)")
    
    logger.info("\n" + "=" * 60)
    logger.info("AGENTIC IDS PIPELINE COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    import numpy as np
    main()
