"""Server registry — tracks live geo-distributed capture servers."""

import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

from .schemas import ServerRegistration

logger = logging.getLogger(__name__)

# Servers are considered dead if no heartbeat in this window
HEARTBEAT_TIMEOUT_SECONDS = 90


class ServerRegistry:
    """
    Thread-safe in-memory registry of all capture servers.

    Sources of truth:
      - Populated from 'server-registry' Kafka topic on startup (consumer replays)
      - Updated in real-time as register / heartbeat / deregister events arrive
      - Directly added via the Server Manager UI (add_server / remove_server)
    """

    # Demo-only presets — these locations have no Docker server yet.
    # MUM (Mumbai), USE (Virginia), LON (London) are handled by Docker and excluded here.
    GEO_PRESETS = [
        {"id": "DEL", "region": "asia-south2",          "label": "Delhi, India",         "lat": 28.704,  "lon": 77.102},
        {"id": "SIN", "region": "asia-southeast1",      "label": "Singapore",             "lat": 1.352,   "lon": 103.820},
        {"id": "TOK", "region": "asia-northeast1",      "label": "Tokyo, Japan",          "lat": 35.689,  "lon": 139.692},
        {"id": "FRA", "region": "europe-west3",         "label": "Frankfurt, Germany",    "lat": 50.110,  "lon": 8.682},
        {"id": "USW", "region": "us-west1",             "label": "Oregon, USA",           "lat": 45.523,  "lon": -122.675},
        {"id": "SYD", "region": "australia-southeast1", "label": "Sydney, Australia",     "lat": -33.868, "lon": 151.209},
        {"id": "SAO", "region": "southamerica-east1",   "label": "São Paulo, Brazil",     "lat": -23.550, "lon": -46.633},
    ]

    def __init__(self):
        self._servers: Dict[str, ServerRegistration] = {}
        self._last_heartbeat: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._partition_counter: int = 0
        
        # Pre-seed the 3 known Docker servers so they always appear immediately
        # in the UI regardless of Kafka heartbeat timing.
        self._preseed_docker_server("MUM-01", "asia-south1", "Mumbai, India", 19.076, 72.877, 0)
        self._preseed_docker_server("USE-01", "us-east1", "Virginia, USA", 38.0, -78.0, 1)
        self._preseed_docker_server("LON-01", "europe-west2", "London, UK", 51.507, -0.127, 2)

    def _preseed_docker_server(self, sid: str, region: str, label: str, lat: float, lon: float, partition: int):
        reg = ServerRegistration(
            server_id=sid, geo_region=region, geo_label=label,
            lat=lat, lon=lon, ip="docker", interface="docker0", event="register"
        )
        reg.kafka_partition = partition
        self._servers[sid] = reg
        # Set heartbeat far in future so it's always considered "alive" for demo
        self._last_heartbeat[sid] = datetime.now(timezone.utc) + timedelta(days=365)
        self._partition_counter = max(self._partition_counter, partition + 1)


    # ─────────────────────────────────────────────────────────
    #  Core CRUD
    # ─────────────────────────────────────────────────────────

    def register(self, reg: ServerRegistration) -> ServerRegistration:
        """Register a new server or refresh an existing one."""
        with self._lock:
            is_new = reg.server_id not in self._servers
            if is_new:
                reg.kafka_partition = self._partition_counter
                self._partition_counter += 1
                logger.info(
                    f"[Registry] NEW server: {reg.server_id} ({reg.geo_label}) "
                    f"→ partition {reg.kafka_partition}"
                )
            else:
                # Preserve partition assignment
                reg.kafka_partition = self._servers[reg.server_id].kafka_partition
            self._servers[reg.server_id] = reg
            self._last_heartbeat[reg.server_id] = datetime.now(timezone.utc)
            return reg

    def heartbeat(self, server_id: str) -> bool:
        """Update last-seen timestamp. Returns False if server not registered."""
        with self._lock:
            if server_id not in self._servers:
                return False
            self._last_heartbeat[server_id] = datetime.now(timezone.utc)
            return True

    def deregister(self, server_id: str) -> bool:
        """Remove a server. Returns False if not found."""
        with self._lock:
            if server_id not in self._servers:
                return False
            del self._servers[server_id]
            del self._last_heartbeat[server_id]
            logger.info(f"[Registry] Deregistered server: {server_id}")
            return True

    # ─────────────────────────────────────────────────────────
    #  Queries
    # ─────────────────────────────────────────────────────────

    def get_all(self) -> List[ServerRegistration]:
        with self._lock:
            return list(self._servers.values())

    def get_active(self) -> List[ServerRegistration]:
        """Servers that sent a heartbeat within HEARTBEAT_TIMEOUT_SECONDS."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)
        with self._lock:
            return [
                s for s in self._servers.values()
                if self._last_heartbeat.get(s.server_id, datetime.min.replace(tzinfo=timezone.utc)) > cutoff
            ]

    def get(self, server_id: str) -> Optional[ServerRegistration]:
        with self._lock:
            return self._servers.get(server_id)

    def get_partition(self, server_id: str) -> int:
        """Return the Kafka partition assigned to a server (-1 if unknown)."""
        with self._lock:
            s = self._servers.get(server_id)
            return s.kafka_partition if s else -1

    def is_alive(self, server_id: str) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)
        with self._lock:
            ts = self._last_heartbeat.get(server_id)
            return ts is not None and ts > cutoff

    def count(self) -> int:
        with self._lock:
            return len(self._servers)

    def to_api_list(self) -> List[dict]:
        """Serialise all servers for the Server Manager API / UI."""
        active_ids = {s.server_id for s in self.get_active()}
        with self._lock:
            result = []
            for s in self._servers.values():
                last_hb = self._last_heartbeat.get(s.server_id)
                result.append({
                    "server_id": s.server_id,
                    "geo_region": s.geo_region,
                    "geo_label": s.geo_label,
                    "lat": s.lat,
                    "lon": s.lon,
                    "ip": s.ip,
                    "interface": s.interface,
                    "event": s.event,
                    "kafka_partition": s.kafka_partition,
                    "registered_at": s.registered_at,
                    "last_heartbeat": last_hb.isoformat() if last_hb else None,
                    "alive": s.server_id in active_ids,
                })
            return result
