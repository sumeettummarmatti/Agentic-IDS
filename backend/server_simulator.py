"""
server_simulator.py
====================
Runs inside each Docker "capture server" container.
Reads its slice of the dataset and replays flows to Kafka,
mimicking a real geo-server sending live network telemetry.

Environment variables (all set by docker-compose):
  SERVER_ID          e.g. MUM-01
  GEO_REGION         e.g. asia-south1
  GEO_LABEL          e.g. "Mumbai, India"
  SLICE_INDEX        0-based index of this server's data slice
  TOTAL_SLICES       total number of server containers
  FLOWS_PER_SECOND   replay rate (0 = burst/max speed)
  DATA_PATH          path to dataset CSV (mounted volume)
  KAFKA_BOOTSTRAP_SERVERS  e.g. kafka:29092
  KAFKA_RAW_FLOWS_TOPIC    e.g. raw-flows
  LOOP               if "true", replay dataset in a loop forever
"""

import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Config from env ────────────────────────────────────────────────────────
SERVER_ID   = os.getenv("SERVER_ID",   "SRV-00")
GEO_REGION  = os.getenv("GEO_REGION",  "unknown")
GEO_LABEL   = os.getenv("GEO_LABEL",   "Unknown")
SLICE_INDEX = int(os.getenv("SLICE_INDEX",  "0"))
TOTAL       = int(os.getenv("TOTAL_SLICES", "1"))
FPS         = float(os.getenv("FLOWS_PER_SECOND", "0"))   # 0 = burst
DATA_PATH   = os.getenv("DATA_PATH",   "/app/data/raw/filtered_nowebatt.csv")
BROKERS     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
TOPIC       = os.getenv("KAFKA_RAW_FLOWS_TOPIC",   "raw-flows")
LOOP        = os.getenv("LOOP", "false").lower() == "true"
# ────────────────────────────────────────────────────────────────────────────


def load_slice(data_path: str, slice_idx: int, total: int):
    """Stratified-sample split: each server gets same attack/benign ratio."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    logger.info(f"Loading dataset from {data_path}…")
    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Drop all-NaN columns
    df = df.dropna(axis=1, how="all")

    label_col = None
    for c in df.columns:
        if c.strip().upper() in ("LABEL", "CLASS", "ATTACK"):
            label_col = c
            break

    if label_col and total > 1:
        # Stratified split so each server has same class distribution
        slices = []
        for label, grp in df.groupby(label_col):
            chunk = len(grp) // total
            slices.append(grp.iloc[slice_idx * chunk : (slice_idx + 1) * chunk])
        return pd.concat(slices).reset_index(drop=True)
    else:
        chunk = len(df) // max(total, 1)
        return df.iloc[slice_idx * chunk : (slice_idx + 1) * chunk].reset_index(drop=True)


def make_producer():
    """Create a confluent-kafka producer connected to BROKERS."""
    from confluent_kafka import Producer

    conf = {
        "bootstrap.servers": BROKERS,
        "client.id": f"aegisflow-sim-{SERVER_ID}",
        "queue.buffering.max.ms": 5,
        "batch.num.messages": 500,
        "request.required.acks": 1,
    }
    return Producer(conf)


def replay(producer, df, fps: float) -> int:
    """Publish each row as a JSON flow record. Returns number published."""
    import pandas as pd

    delay = 1.0 / fps if fps > 0 else 0.0
    published = 0
    t_batch = time.perf_counter()

    from src.kafka.schemas import RawFlowMessage
    import uuid

    for i, row in df.iterrows():
        # Build payload features (drop Label to prevent leaking into pipeline)
        raw_dict = row.to_dict()
        features = {k: v for k, v in raw_dict.items() if k.strip().upper() not in ("LABEL", "CLASS", "ATTACK")}
        
        # Extract Flow ID or construct one
        flow_id = raw_dict.get("Flow ID", f"{SERVER_ID}-{i}")

        # Sanitize: replace inf/nan with 0
        clean_features = {}
        for k, v in features.items():
            if isinstance(v, float) and not math.isfinite(v):
                clean_features[k] = 0.0
            else:
                clean_features[k] = v

        msg = RawFlowMessage(
            server_id=SERVER_ID,
            geo_region=GEO_REGION,
            flow_id=flow_id,
            features=clean_features,
            message_id=str(uuid.uuid4())
        )

        payload = msg.to_json().encode("utf-8")

        # Produce keyed by SERVER_ID → same partition every time
        producer.produce(
            topic=TOPIC,
            key=SERVER_ID.encode(),
            value=payload,
        )
        published += 1

        if delay > 0:
            time.sleep(delay)

        # Poll every 100 messages to serve delivery reports
        if published % 100 == 0:
            producer.poll(0)
            elapsed = time.perf_counter() - t_batch
            actual_fps = 100 / elapsed if elapsed > 0 else 0
            logger.info(f"[{SERVER_ID}] {published}/{len(df)} flows  "
                        f"({actual_fps:.0f} flows/s)")
            t_batch = time.perf_counter()

    producer.flush(timeout=10)
    return published


def wait_for_kafka(brokers: str, retries: int = 12, interval: int = 5):
    """Block until Kafka is reachable (retry loop for container startup race)."""
    from confluent_kafka.admin import AdminClient

    logger.info(f"Waiting for Kafka at {brokers}…")
    for attempt in range(1, retries + 1):
        try:
            admin = AdminClient({"bootstrap.servers": brokers})
            meta = admin.list_topics(timeout=3)
            logger.info(f"Kafka is ready (attempt {attempt}): "
                        f"{len(meta.topics)} topics")
            return
        except Exception as exc:
            logger.warning(f"  Attempt {attempt}/{retries}: {exc} — retrying in {interval}s")
            time.sleep(interval)
    raise RuntimeError(f"Kafka at {brokers} not reachable after {retries} attempts")


def main():
    logger.info("=" * 60)
    logger.info(f"  AegisFlow Server Simulator")
    logger.info(f"  ID: {SERVER_ID}  Region: {GEO_REGION}  ({GEO_LABEL})")
    logger.info(f"  Slice {SLICE_INDEX + 1}/{TOTAL}  |  {FPS or 'burst'} flows/s")
    logger.info(f"  Kafka: {BROKERS} → {TOPIC}")
    logger.info("=" * 60)

    # Wait for Kafka to be ready (important for container startup ordering)
    wait_for_kafka(BROKERS)

    # Load this server's slice of the dataset
    df = load_slice(DATA_PATH, SLICE_INDEX, TOTAL)
    logger.info(f"Loaded {len(df)} flows for {SERVER_ID}")

    producer = make_producer()
    run = 0

    while True:
        run += 1
        logger.info(f"[{SERVER_ID}] Starting replay run #{run}…")
        t0 = time.time()
        n = replay(producer, df, FPS)
        elapsed = time.time() - t0
        logger.info(
            f"[{SERVER_ID}] Run #{run} complete: "
            f"{n} flows in {elapsed:.1f}s  "
            f"({n/elapsed:.0f} flows/s avg)"
        )

        if not LOOP:
            logger.info(f"[{SERVER_ID}] LOOP=false — exiting after one pass.")
            break

        logger.info(f"[{SERVER_ID}] LOOP=true — restarting in 5s…")
        time.sleep(5)


if __name__ == "__main__":
    main()
