"""
AegisFlow → IDS Preprocessor column bridge.

The C engine (AegisFlow) exports ~44 flow features as snake_case JSON keys
(see AegisFlow/src/exporter/json_exporter.c). The Preprocessor / models were fitted on
CICFlowMeter-style column names ("Total Fwd Packet", "Flow Bytes/s", ...).

This module:
  1. renames the AegisFlow keys to the training column names,
  2. derives a few training columns that are simple functions of exported ones
     (variance, segment size, subflow counts, per-direction packet rate, ...).

Columns AegisFlow does not compute at all (header lengths, directional PSH/URG flags,
Fwd Act Data Pkts, Fwd Seg Size Min, Active*) are NOT invented here. The ingestion stage
zero-fills anything the fitted preprocessor expects but the message lacks, so live-capture
predictions rely on those columns being 0 — treat accuracy on real traffic as unvalidated.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  AegisFlow JSON key → training column name (direct renames)
# ─────────────────────────────────────────────────────────────
AEGISFLOW_COLUMN_MAP: Dict[str, str] = {
    "protocol":               "Protocol",
    "flow_duration":          "Flow Duration",      # microseconds in both
    # Packet / byte counts
    "total_fwd_packets":      "Total Fwd Packet",
    "total_bwd_packets":      "Total Bwd packets",
    "total_length_fwd_pkts":  "Total Length of Fwd Packet",
    "total_length_bwd_pkts":  "Total Length of Bwd Packet",
    # Rates
    "flow_bytes_per_sec":     "Flow Bytes/s",
    "flow_pkts_per_sec":      "Flow Packets/s",
    # Packet length (both directions combined)
    "pkt_length_min":         "Packet Length Min",
    "pkt_length_max":         "Packet Length Max",
    "pkt_length_mean":        "Packet Length Mean",
    "pkt_length_std":         "Packet Length Std",
    # Fwd packet length
    "fwd_pkt_length_max":     "Fwd Packet Length Max",
    "fwd_pkt_length_min":     "Fwd Packet Length Min",
    "fwd_pkt_length_mean":    "Fwd Packet Length Mean",
    "fwd_pkt_length_std":     "Fwd Packet Length Std",
    # Bwd packet length
    "bwd_pkt_length_max":     "Bwd Packet Length Max",
    "bwd_pkt_length_min":     "Bwd Packet Length Min",
    "bwd_pkt_length_mean":    "Bwd Packet Length Mean",
    "bwd_pkt_length_std":     "Bwd Packet Length Std",
    # Flow IAT
    "flow_iat_mean":          "Flow IAT Mean",
    "flow_iat_std":           "Flow IAT Std",
    "flow_iat_max":           "Flow IAT Max",
    "flow_iat_min":           "Flow IAT Min",
    # Fwd IAT
    "fwd_iat_mean":           "Fwd IAT Mean",
    "fwd_iat_std":            "Fwd IAT Std",
    "fwd_iat_max":            "Fwd IAT Max",
    "fwd_iat_min":            "Fwd IAT Min",
    # Bwd IAT
    "bwd_iat_mean":           "Bwd IAT Mean",
    "bwd_iat_std":            "Bwd IAT Std",
    "bwd_iat_max":            "Bwd IAT Max",
    "bwd_iat_min":            "Bwd IAT Min",
    # TCP flag counts
    "fin_flag_count":         "FIN Flag Count",
    "syn_flag_count":         "SYN Flag Count",
    "rst_flag_count":         "RST Flag Count",
    "psh_flag_count":         "PSH Flag Count",
    "ack_flag_count":         "ACK Flag Count",
    "urg_flag_count":         "URG Flag Count",
}


def is_aegisflow_record(features: Dict[str, Any]) -> bool:
    """True if this feature dict came from the C engine rather than a CSV replay."""
    return "total_fwd_packets" in features


def _num(d: Dict[str, Any], key: str) -> float:
    try:
        v = float(d.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def bridge_feature_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one AegisFlow feature dict into a dict keyed by training column names.

    Unmapped keys (src_ip, dst_port, ...) are kept as-is; the ingestion stage later
    reindexes to the fitted column list, which drops them.
    """
    out: Dict[str, Any] = {AEGISFLOW_COLUMN_MAP.get(k, k): v for k, v in raw.items()}

    fwd_pkts = _num(raw, "total_fwd_packets")
    bwd_pkts = _num(raw, "total_bwd_packets")
    fwd_bytes = _num(raw, "total_length_fwd_pkts")
    bwd_bytes = _num(raw, "total_length_bwd_pkts")
    total_pkts = fwd_pkts + bwd_pkts
    duration_s = _num(raw, "flow_duration") / 1e6        # exported in microseconds

    # Derived: exact (or standard CICFlowMeter) functions of exported values
    out["Packet Length Variance"] = _num(raw, "pkt_length_std") ** 2
    out["Fwd Segment Size Avg"] = _num(raw, "fwd_pkt_length_mean")
    out["Bwd Segment Size Avg"] = _num(raw, "bwd_pkt_length_mean")
    out["Subflow Fwd Packets"] = fwd_pkts
    out["Subflow Fwd Bytes"] = fwd_bytes
    out["Subflow Bwd Packets"] = bwd_pkts
    out["Subflow Bwd Bytes"] = bwd_bytes
    out["Average Packet Size"] = (fwd_bytes + bwd_bytes) / total_pkts if total_pkts else 0.0
    out["Down/Up Ratio"] = float(int(bwd_pkts / fwd_pkts)) if fwd_pkts else 0.0
    out["Fwd IAT Total"] = _num(raw, "fwd_iat_mean") * max(fwd_pkts - 1, 0)
    out["Bwd IAT Total"] = _num(raw, "bwd_iat_mean") * max(bwd_pkts - 1, 0)
    out["Fwd Packets/s"] = fwd_pkts / duration_s if duration_s > 0 else 0.0
    out["Bwd Packets/s"] = bwd_pkts / duration_s if duration_s > 0 else 0.0
    return out


def bridge_features(raw_features: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert an AegisFlow JSON feature dict → single-row numeric DataFrame with
    training column names. NaN / inf are replaced with 0.
    """
    df = pd.DataFrame([bridge_feature_dict(raw_features)])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0.0)


def bridge_features_from_csv_row(row: pd.Series) -> pd.DataFrame:
    """
    Convert a row from our existing CSV dataset → DataFrame.
    The CSV already has the training column names, so no renaming is needed.
    """
    d = row.to_dict()
    d.pop("Label", None)
    d.pop("label", None)
    df = pd.DataFrame([d])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0.0)
