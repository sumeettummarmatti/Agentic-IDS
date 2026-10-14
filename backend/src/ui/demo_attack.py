"""
Attack Lab helper — injects a labelled burst of real dataset flows into the
'raw-flows' Kafka topic so the full pipeline (ML -> council -> PPO) handles it
exactly like production traffic.

Injected flows carry flow_id "DEMO-<run_id>-<n>" so the simulation page can
follow just these flows in the shared live event stream.
"""

import csv
import json
import logging
import os
import random
import threading
import time
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# UI scenario -> dataset Label
SCENARIOS = {
    "normal": "BENIGN",
    "ddos": "DDoS",
    "portscan": "PortScan",
}

# Same locations the docker-compose simulators use; lets the Attack Lab work even when
# the simulators are stopped and their registry messages have expired from Kafka.
DEFAULT_LOCATIONS = [
    {"server_id": "MUM-01", "geo_region": "asia-south1", "geo_label": "Mumbai, India",
     "lat": 19.076, "lon": 72.877},
    {"server_id": "USE-01", "geo_region": "us-east4", "geo_label": "Virginia, USA",
     "lat": 38.946, "lon": -77.447},
    {"server_id": "LON-01", "geo_region": "europe-west2", "geo_label": "London, UK",
     "lat": 51.507, "lon": -0.128},
]

_SAMPLES_PER_LABEL = 300
_samples: Dict[str, List[dict]] = {}
_samples_lock = threading.Lock()

_NON_FEATURE = {"Label"}


def _dataset_path() -> str:
    for p in (
        os.getenv("DEMO_DATASET", ""),
        "data/raw/filtered_nowebatt.csv",
        "/app/data/raw/filtered_nowebatt.csv",
    ):
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("filtered_nowebatt.csv not found (set DEMO_DATASET)")


def _to_value(v: str):
    try:
        return float(v)
    except ValueError:
        return v


def _load_samples():
    """Stream the CSV once, keeping the first N rows of each label."""
    with _samples_lock:
        if _samples:
            return
        wanted = set(SCENARIOS.values())
        found: Dict[str, List[dict]] = {k: [] for k in wanted}
        with open(_dataset_path(), newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            li = header.index("Label")
            for row in reader:
                lab = row[li]
                if lab in found and len(found[lab]) < _SAMPLES_PER_LABEL:
                    found[lab].append(
                        {h: _to_value(v) for h, v in zip(header, row) if h not in _NON_FEATURE}
                    )
                if all(len(v) >= _SAMPLES_PER_LABEL for v in found.values()):
                    break
        _samples.update(found)
        logger.info(f"[Demo] Loaded samples: { {k: len(v) for k, v in found.items()} }")


def launch_attack(scenario: str, server: dict, count: int = 30, rate: float = 4.0) -> dict:
    """
    Publish `count` flows of `scenario` for `server` (a registry dict) in a
    background thread. Returns immediately with the run id.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    _load_samples()
    rows = _samples[SCENARIOS[scenario]]
    run_id = uuid.uuid4().hex[:6]
    server_id = server["server_id"]
    geo_region = server.get("geo_region", "unknown")
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_RAW_FLOWS_TOPIC", "raw-flows")

    def _run():
        from confluent_kafka import Producer
        from src.kafka.schemas import RawFlowMessage

        producer = Producer({"bootstrap.servers": bootstrap})
        picks = random.sample(rows, min(count, len(rows)))
        for n, feats in enumerate(picks):
            msg = RawFlowMessage(
                server_id=server_id,
                geo_region=geo_region,
                flow_id=f"DEMO-{run_id}-{n}",
                features=feats,
            )
            producer.produce(topic, key=server_id.encode(), value=msg.to_json().encode())
            producer.poll(0)
            if rate > 0:
                time.sleep(1.0 / rate)
        producer.flush(10)
        logger.info(f"[Demo] Injected {len(picks)} {scenario} flows into {server_id} (run {run_id})")

    threading.Thread(target=_run, daemon=True, name=f"demo-{run_id}").start()
    return {"run_id": run_id, "scenario": scenario, "server_id": server_id, "count": count,
            "flow_prefix": f"DEMO-{run_id}-"}
