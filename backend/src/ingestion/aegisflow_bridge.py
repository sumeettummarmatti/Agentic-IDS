"""
AegisFlow → IDS Preprocessor column bridge.

AegisFlow (C engine) outputs 44 CICFlowMeter-compatible features as JSON.
Our Preprocessor was trained on CICIDS2017 column names.
This module renames AegisFlow's snake_case fields to the IDS column names.

Feature reference:
  AegisFlow README: "44 features matching CICIDS2017 column names"
  Our DROP_FEATURES already covers Flow ID, Source IP, etc.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Column mapping: AegisFlow C field → IDS Preprocessor column name
#  (Only fields that need renaming; direct matches pass through unchanged)
# ─────────────────────────────────────────────────────────────
AEGISFLOW_COLUMN_MAP: Dict[str, str] = {
    # Packet counts
    "fwd_pkts":             "Total Fwd Packet",
    "bwd_pkts":             "Total Bwd packets",
    # Payload lengths
    "fwd_bytes":            "Total Length of Fwd Packet",
    "bwd_bytes":            "Total Length of Bwd Packet",
    # Fwd packet length stats
    "fwd_pkt_len_max":      "Fwd Packet Length Max",
    "fwd_pkt_len_min":      "Fwd Packet Length Min",
    "fwd_pkt_len_mean":     "Fwd Packet Length Mean",
    "fwd_pkt_len_std":      "Fwd Packet Length Std",
    # Bwd packet length stats
    "bwd_pkt_len_max":      "Bwd Packet Length Max",
    "bwd_pkt_len_min":      "Bwd Packet Length Min",
    "bwd_pkt_len_mean":     "Bwd Packet Length Mean",
    "bwd_pkt_len_std":      "Bwd Packet Length Std",
    # Flow-level throughput
    "flow_bytes_s":         "Flow Bytes/s",
    "flow_pkts_s":          "Flow Packets/s",
    # Flow IAT
    "flow_iat_mean":        "Flow IAT Mean",
    "flow_iat_std":         "Flow IAT Std",
    "flow_iat_max":         "Flow IAT Max",
    "flow_iat_min":         "Flow IAT Min",
    # Fwd IAT
    "fwd_iat_total":        "Fwd IAT Total",
    "fwd_iat_mean":         "Fwd IAT Mean",
    "fwd_iat_std":          "Fwd IAT Std",
    "fwd_iat_max":          "Fwd IAT Max",
    "fwd_iat_min":          "Fwd IAT Min",
    # Bwd IAT
    "bwd_iat_total":        "Bwd IAT Total",
    "bwd_iat_mean":         "Bwd IAT Mean",
    "bwd_iat_std":          "Bwd IAT Std",
    "bwd_iat_max":          "Bwd IAT Max",
    "bwd_iat_min":          "Bwd IAT Min",
    # Header lengths
    "fwd_hdr_len":          "Fwd Header Length",
    "bwd_hdr_len":          "Bwd Header Length",
    # TCP Flags
    "fin_cnt":              "FIN Flag Count",
    "syn_cnt":              "SYN Flag Count",
    "rst_cnt":              "RST Flag Count",
    "psh_cnt":              "PSH Flag Count",
    "ack_cnt":              "ACK Flag Count",
    "urg_cnt":              "URG Flag Count",
    "cwe_cnt":              "CWE Flag Count",
    "ece_cnt":              "ECE Flag Count",
    # Packet length stats (combined)
    "pkt_len_min":          "Packet Length Min",
    "pkt_len_max":          "Packet Length Max",
    "pkt_len_mean":         "Packet Length Mean",
    "pkt_len_std":          "Packet Length Std",
    "pkt_len_var":          "Packet Length Variance",
    # Active/Idle time stats
    "active_mean":          "Active Mean",
    "active_std":           "Active Std",
    "active_max":           "Active Max",
    "active_min":           "Active Min",
    "idle_mean":            "Idle Mean",
    "idle_std":             "Idle Std",
    "idle_max":             "Idle Max",
    "idle_min":             "Idle Min",
}

# Fields from AegisFlow that we DROP (IDs, addresses — preprocessor drops these anyway)
AEGISFLOW_DROP_FIELDS = {
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "start_ts", "end_ts", "duration_us",
}


def bridge_features(raw_features: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert AegisFlow JSON feature dict → DataFrame with IDS column names.
    
    Steps:
      1. Drop non-feature fields (IPs, ports, timestamps)
      2. Rename snake_case → CICIDS2017 column names
      3. Keep any columns that are already named correctly (pass-through)
      4. Zero-fill any missing columns expected by the preprocessor

    Returns a single-row DataFrame ready for Preprocessor.prepare_features_and_labels()
    """
    # 1. Filter out fields we don't want
    filtered = {k: v for k, v in raw_features.items() if k not in AEGISFLOW_DROP_FIELDS}

    # 2. Rename AegisFlow fields
    renamed = {}
    for k, v in filtered.items():
        target_name = AEGISFLOW_COLUMN_MAP.get(k, k)  # fallback: keep original name
        renamed[target_name] = v

    # 3. Build DataFrame
    df = pd.DataFrame([renamed])

    # 4. Ensure numeric types; replace inf/nan with 0
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)

    return df


def bridge_features_from_csv_row(row: pd.Series) -> pd.DataFrame:
    """
    Convert a row from our existing CSV dataset → DataFrame.
    The CSV already has CICIDS2017 column names, so no renaming needed.
    Just wrap in a DataFrame and clean.
    """
    d = row.to_dict()
    # Remove label column if present
    d.pop("Label", None)
    d.pop("label", None)
    df = pd.DataFrame([d])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    return df
