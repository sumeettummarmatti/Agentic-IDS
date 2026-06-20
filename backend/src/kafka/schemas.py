"""
Kafka message schemas for Agentic-IDS.
All messages are JSON-serializable dataclasses with a to_dict() / from_dict() pair.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json
import uuid


# ─────────────────────────────────────────────────────────────
#  Server Registration
# ─────────────────────────────────────────────────────────────

@dataclass
class ServerRegistration:
    """
    Published to the 'server-registry' topic when a capture server
    comes online, sends a heartbeat, or goes offline.
    """
    server_id: str                  # Unique ID e.g. "MUM-01"
    geo_region: str                 # e.g. "asia-south1"
    geo_label: str                  # Human label e.g. "Mumbai, India"
    lat: float                      # Latitude  (for map display)
    lon: float                      # Longitude (for map display)
    ip: str                         # Server IP
    interface: str                  # Network interface being captured ("eth0")
    event: str = "register"         # "register" | "heartbeat" | "deregister"
    aegisflow_version: str = "feat/80plus_feature"
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    kafka_partition: int = -1       # Assigned partition (-1 = not yet assigned)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "ServerRegistration":
        return cls(**json.loads(data))


# ─────────────────────────────────────────────────────────────
#  Raw Flow (from AegisFlow or CSV replay)
# ─────────────────────────────────────────────────────────────

@dataclass
class RawFlowMessage:
    """
    Published to 'raw-flows' topic.
    One message = one completed network flow (bidirectional, timed out or FIN).
    """
    server_id: str                  # Origin server
    geo_region: str                 # Origin geo
    flow_id: str                    # "src_ip:port-dst_ip:port-proto"
    features: Dict[str, Any]        # 44 CICFlowMeter-compatible feature fields
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> str:
        return json.dumps({
            "server_id": self.server_id,
            "geo_region": self.geo_region,
            "flow_id": self.flow_id,
            "features": self.features,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        })

    @classmethod
    def from_json(cls, data: str) -> "RawFlowMessage":
        d = json.loads(data)
        return cls(**d)


# ─────────────────────────────────────────────────────────────
#  Threat Alert (Detection → Council)
# ─────────────────────────────────────────────────────────────

@dataclass
class ThreatAlertMessage:
    """
    Published to 'threats' topic when ensemble detection confidence > threshold.
    Also consumed internally by ThreatCouncilStage.
    """
    server_id: str
    geo_region: str
    flow_id: str
    attack_type: str                # "DDoS" | "PortScan" | "BENIGN" | ...
    confidence: float
    severity: str                   # "High" | "Medium" | "Low"
    raw_features: Dict[str, Any]
    council_report: Optional[Dict] = None     # filled in after council runs
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "ThreatAlertMessage":
        return cls(**json.loads(data))


# ─────────────────────────────────────────────────────────────
#  Defense Decision (Defender → decisions topic)
# ─────────────────────────────────────────────────────────────

@dataclass
class DefenseDecisionMessage:
    """
    Published to 'decisions' topic — the final output of the IDS pipeline.
    Audit log of every defensive action taken.
    """
    server_id: str
    geo_region: str
    flow_id: str
    attack_type: str
    confidence: float
    action: str                     # "BLOCK_SOURCE" | "RATE_LIMIT" | "DEEP_PACKET_INSPECTION" | "MONITOR"
    action_id: int
    status: str                     # "executed" | "failed"
    latency_ms: float               # End-to-end pipeline latency for this flow
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "DefenseDecisionMessage":
        return cls(**json.loads(data))
