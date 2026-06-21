import os
import argparse
import asyncio
import logging
import sys

import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv(override=True)
load_dotenv(dotenv_path='config/.env', override=True)

# Support both env var names (PRIMARY_LLM_PROVIDER from existing .env, LLM_PROVIDER from new)
def _get_provider_from_env() -> str:
    return (
        os.getenv('LLM_PROVIDER')
        or os.getenv('PRIMARY_LLM_PROVIDER')
        or 'groq'
    ).lower()


from src.detector.ensemble_model import EnsembleDetector
from src.detector.preprocessor import Preprocessor
from src.agents.attacker_agents import generate_balanced_synthetic_dataset
from src.agents.defender_agent import DefenderRLAgent
from src.council.llm_council_wrapper import ThreatAnalysisCouncil

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Agentic IDS')
    # ── Mode ──────────────────────────────────────────────
    parser.add_argument(
        '--mode', choices=['test', 'production', 'ui'],
        default='test',
        help=(
            'test       : process existing dataset, no Kafka required\n'
            'production : read live flows from Kafka (requires docker-compose up)\n'
            'ui         : launch Server Manager dashboard + run pipeline'
        )
    )
    # ── Test mode options ──────────────────────────────────
    parser.add_argument('--data', type=str,
                        default='data/raw/filtered_nowebatt.csv',
                        help='Path to dataset CSV/XLSX (test mode)')
    parser.add_argument('--n-servers', type=int, default=3,
                        help='Number of virtual geo-servers to simulate (test mode)')
    parser.add_argument('--fps', type=float, default=0,
                        help='Flows per second per server. 0 = burst (test mode)')
    # ── Provider ───────────────────────────────────────────
    parser.add_argument('--provider', choices=['hf', 'groq', 'ollama'],
                        help='Override LLM_PROVIDER from .env')
    # ── Legacy compat ──────────────────────────────────────
    parser.add_argument('--live-data', type=str,
                        help='(Legacy) Alias for --data --mode test')
    return parser.parse_args()



def _all_models_exist() -> bool:
    """Return True only if every required model file is present on disk."""
    from pathlib import Path
    required = [
        Path('models/trained/xgboost_multiclass.pkl'),
        Path('models/trained/class_mapping.pkl'),
        Path('models/trained/lstm_detector.pt'),
        Path('models/trained/preprocessor.pkl'),
        Path('models/defender_ppo.zip'),
    ]
    return all(p.exists() for p in required)


def _load_components(provider: str):
    """Fast path: load all trained artefacts from disk — no CSV read, no training."""
    logger.info("[PHASE 1] Loading saved models from disk (fast path)…")

    preprocessor = Preprocessor.load('models/trained/preprocessor.pkl')
    detector     = EnsembleDetector(use_lstm=True)
    # Sync the scaler from the preprocessor into the detector (used by predict())
    detector.scaler = preprocessor.scaler
    council  = ThreatAnalysisCouncil(provider=provider)
    # DefenderRLAgent._initialize_model() auto-loads models/defender_ppo.zip if present
    defender = DefenderRLAgent()

    logger.info("✓ All components loaded from disk\n")
    return detector, council, defender, preprocessor


def _train_components(provider: str):
    """Full training path: read CSV, fit preprocessor, train XGBoost + LSTM, save everything."""
    logger.info("[PHASE 1] Training IDS components from scratch…")

    detector     = EnsembleDetector(use_lstm=True)
    council      = ThreatAnalysisCouncil(provider=provider)
    defender     = DefenderRLAgent()
    preprocessor = Preprocessor()

    user_data_path    = 'data/raw/filtered_nowebatt.csv'
    default_data_path = 'data/raw/Darknet.xlsx'
    processed_dir     = 'data/processed'

    # ── Try processed numpy cache first (skips 117 MB CSV re-read) ──────────
    import os
    cache_X = os.path.join(processed_dir, 'X_train.npy')
    cache_y = os.path.join(processed_dir, 'y_train.npy')

    if os.path.exists(cache_X) and os.path.exists(cache_y):
        logger.info(f"Loading preprocessed cache from {processed_dir}…")
        X_train = np.load(cache_X)
        y_train = np.load(cache_y)
        # Restore preprocessor fit state from saved pkl if it exists (partial save)
        pp_path = 'models/trained/preprocessor.pkl'
        if os.path.exists(pp_path):
            preprocessor = Preprocessor.load(pp_path)
        else:
            logger.warning("Preprocessor pkl missing — will refit from cache arrays (no feature names)")
    else:
        # ── Load raw CSV / XLSX ─────────────────────────────────────────────
        if os.path.exists(user_data_path):
            logger.info(f"Loading training data from {user_data_path}…")
            df = preprocessor.load_data(user_data_path)
            X, _, y = preprocessor.prepare_features_and_labels(df, training=True)
        else:
            logger.info(f"Loading training data from {default_data_path}…")
            df = preprocessor.load_data(default_data_path)
            X_real, _, y_real = preprocessor.prepare_features_and_labels(df, training=True)
            synthetic_df = generate_balanced_synthetic_dataset(num_ddos=500, num_portscan=200)
            required_columns = preprocessor.feature_names or []
            for col in required_columns:
                if col not in synthetic_df.columns:
                    synthetic_df[col] = 0
            X_syn = synthetic_df[required_columns].values if required_columns else synthetic_df.values
            y_syn = np.zeros(len(synthetic_df))
            X = np.vstack([X_real, X_syn])
            y = np.hstack([y_real, y_syn])

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # ── Save preprocessed numpy cache so next run is fast ───────────────
        os.makedirs(processed_dir, exist_ok=True)
        np.save(cache_X, X_train)
        np.save(cache_y, y_train)
        logger.info(f"✓ Saved processed arrays to {processed_dir}/")

        # ── Save fitted preprocessor ────────────────────────────────────────
        preprocessor.save('models/trained/preprocessor.pkl')

    # ── Train models (XGBoost + LSTM are saved inside detector.train()) ─────
    detector.scaler = preprocessor.scaler
    detector.train(X_train, y_train)
    defender.train(total_timesteps=500)

    logger.info("✓ All components trained and saved\n")
    return detector, council, defender, preprocessor


def build_components(provider: str):
    """Initialise all IDS components — loads from disk if possible, trains otherwise."""
    if _all_models_exist():
        return _load_components(provider)
    return _train_components(provider)




# ─────────────────────────────────────────────────────────────
#  Mode: TEST
# ─────────────────────────────────────────────────────────────

def run_test(args, detector, council, defender, preprocessor):
    from src.pipeline.pipeline import IDSPipeline

    logger.info(f"\n[PHASE 2] TEST MODE — {args.n_servers} virtual servers | {args.data}")
    pipeline = IDSPipeline(detector, council, defender, preprocessor, mode='test')

    report = asyncio.run(pipeline.run_test(
        data_path=args.data,
        n_servers=args.n_servers,
        flows_per_second_per_server=args.fps,
    ))

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    for k, v in report.items():
        logger.info(f"  {k:<26} {v}")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────
#  Mode: PRODUCTION
# ─────────────────────────────────────────────────────────────

def run_production(args, detector, council, defender, preprocessor):
    from src.pipeline.pipeline import IDSPipeline
    from src.kafka.registry import ServerRegistry
    from src.kafka.consumer import ServerRegistryConsumer

    logger.info("\n[PHASE 2] PRODUCTION MODE — Listening on Kafka…")
    logger.info("  Start AegisFlow capture server(s) to send flows.")
    logger.info("  Open http://localhost:5001 to manage servers.\n")

    registry = ServerRegistry()
    reg_consumer = ServerRegistryConsumer(registry)
    reg_consumer.start()

    pipeline = IDSPipeline(detector, council, defender, preprocessor, mode='production')

    try:
        asyncio.run(pipeline.run_production(registry))
    except KeyboardInterrupt:
        logger.info("\n[PRODUCTION] Shutting down…")
    finally:
        reg_consumer.stop()


# ─────────────────────────────────────────────────────────────
#  Mode: UI (Server Manager Dashboard)
# ─────────────────────────────────────────────────────────────

def run_ui(args, detector, council, defender, preprocessor):
    import threading
    from src.pipeline.pipeline import IDSPipeline
    from src.kafka.registry import ServerRegistry
    from src.kafka.consumer import ServerRegistryConsumer
    import src.ui.server_manager as sm

    logger.info("\n[PHASE 2] UI MODE — Server Manager at http://localhost:5001")

    pipeline = IDSPipeline(detector, council, defender, preprocessor, mode='production')
    sm.registry = ServerRegistry()
    sm.pipeline = pipeline  # expose to monitor/unmonitor API endpoints

    # Start registry consumer so Docker geo-servers appear on the map automatically
    reg_consumer = ServerRegistryConsumer(sm.registry)
    reg_consumer.start()

    # Background: sync pipeline stats to UI module
    def sync_stats():
        import time
        while True:
            sm.pipeline_stats = pipeline.live_stats()
            time.sleep(2)

    threading.Thread(target=sync_stats, daemon=True).start()

    # Background: run the actual Kafka consumer pipeline
    def run_pipeline():
        asyncio.run(pipeline.run_production(sm.registry))
        
    threading.Thread(target=run_pipeline, daemon=True).start()

    try:
        import uvicorn
        uvicorn.run(sm.app, host='0.0.0.0', port=5001, log_level='warning')
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn fastapi")
        sys.exit(1)
    finally:
        reg_consumer.stop()


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Legacy compat
    if args.live_data:
        args.data = args.live_data
        args.mode = 'test'

    provider = args.provider or _get_provider_from_env()
    os.environ['LLM_PROVIDER'] = provider

    logger.info("=" * 70)
    logger.info("  AGENTIC-IDS  |  Multi-Agent Intrusion Detection System")
    logger.info(f"  Mode: {args.mode.upper()}  |  LLM: {provider.upper()}")
    logger.info("=" * 70)

    detector, council, defender, preprocessor = build_components(provider)

    if args.mode == 'test':
        run_test(args, detector, council, defender, preprocessor)
    elif args.mode == 'production':
        run_production(args, detector, council, defender, preprocessor)
    elif args.mode == 'ui':
        run_ui(args, detector, council, defender, preprocessor)


if __name__ == '__main__':
    main()
