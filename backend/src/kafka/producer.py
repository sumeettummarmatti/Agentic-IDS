"""
Kafka producer for Agentic-IDS.

FlowProducer  — publishes RawFlowMessage to 'raw-flows' topic, keyed on server_id
                so all flows from the same server land on the same partition.
                Also handles server registration + heartbeats to 'server-registry'.
"""

import json
import logging
import threading
import time
import os
from typing import Optional

from .schemas import RawFlowMessage, ServerRegistration

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 30


class FlowProducer:
    """
    Wraps confluent_kafka.Producer with:
      - Automatic server registration on startup
      - Periodic heartbeat thread
      - Partition-keyed flow publishing (server_id → same partition always)
      - Graceful shutdown

    Usage (production / simulation):
        producer = FlowProducer(server_id="MUM-01", geo_region="asia-south1", ...)
        producer.start()
        producer.publish_flow(features_dict)
        producer.stop()
    """

    def __init__(
        self,
        server_id: str,
        geo_region: str,
        geo_label: str,
        lat: float,
        lon: float,
        ip: str,
        interface: str = "eth0",
        bootstrap_servers: Optional[str] = None,
        raw_flows_topic: Optional[str] = None,
        registry_topic: Optional[str] = None,
    ):
        self.server_id = server_id
        self.geo_region = geo_region
        self.geo_label = geo_label
        self.lat = lat
        self.lon = lon
        self.ip = ip
        self.interface = interface

        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.raw_flows_topic = raw_flows_topic or os.getenv(
            "KAFKA_RAW_FLOWS_TOPIC", "raw-flows"
        )
        self.registry_topic = registry_topic or os.getenv(
            "KAFKA_SERVER_REGISTRY_TOPIC", "server-registry"
        )

        self._producer = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False
        self._published_count = 0

    # ─────────────────────────────────────────────────────────
    #  Lifecycle
    # ─────────────────────────────────────────────────────────

    def start(self):
        """Connect to Kafka, register server, start heartbeat thread."""
        try:
            from confluent_kafka import Producer
            self._producer = Producer({
                "bootstrap.servers": self.bootstrap_servers,
                "client.id": f"agentic-ids-producer-{self.server_id}",
                # Batching for throughput (industry default)
                "linger.ms": 5,
                "batch.size": 65536,
                "compression.type": "lz4",
                "acks": "1",          # Leader ack only — good tradeoff for IDS
            })
            logger.info(f"[Producer:{self.server_id}] Connected to Kafka at {self.bootstrap_servers}")
        except ImportError:
            logger.warning("[Producer] confluent_kafka not installed — running in NO-OP mode")
            self._producer = None

        self._running = True
        self._register(event="register")

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name=f"heartbeat-{self.server_id}"
        )
        self._heartbeat_thread.start()

    def stop(self):
        """Flush pending messages, deregister server, stop heartbeat."""
        self._running = False
        self._register(event="deregister")
        if self._producer:
            self._producer.flush(timeout=10)
        logger.info(f"[Producer:{self.server_id}] Stopped. Total published: {self._published_count}")

    # ─────────────────────────────────────────────────────────
    #  Publishing
    # ─────────────────────────────────────────────────────────

    def publish_flow(self, features: dict, flow_id: str = "unknown") -> bool:
        """
        Publish one completed flow to the 'raw-flows' topic.
        Partition key = server_id → guaranteed same partition.
        Returns True on success.
        """
        msg = RawFlowMessage(
            server_id=self.server_id,
            geo_region=self.geo_region,
            flow_id=flow_id,
            features=features,
        )

        if not self._producer:
            # No-op / test mode — just count
            self._published_count += 1
            return True

        try:
            self._producer.produce(
                topic=self.raw_flows_topic,
                key=self.server_id.encode("utf-8"),   # partition key
                value=msg.to_json().encode("utf-8"),
                on_delivery=self._delivery_callback,
            )
            # Poll every 10 messages to process delivery reports promptly.
            # Original was every 100 — fine for throughput but delays error detection.
            if self._published_count % 10 == 0:
                self._producer.poll(0)
            self._published_count += 1
            return True
        except Exception as e:
            logger.error(f"[Producer:{self.server_id}] Failed to publish flow: {e}")
            return False

    # ─────────────────────────────────────────────────────────
    #  Registration & Heartbeat
    # ─────────────────────────────────────────────────────────

    def _register(self, event: str = "register"):
        reg = ServerRegistration(
            server_id=self.server_id,
            geo_region=self.geo_region,
            geo_label=self.geo_label,
            lat=self.lat,
            lon=self.lon,
            ip=self.ip,
            interface=self.interface,
            event=event,
        )
        payload = reg.to_json().encode("utf-8")

        if not self._producer:
            logger.info(f"[Producer:{self.server_id}] [NO-OP] Registration event: {event}")
            return

        try:
            self._producer.produce(
                topic=self.registry_topic,
                key=self.server_id.encode("utf-8"),
                value=payload,
            )
            self._producer.flush(timeout=5)
            logger.info(f"[Producer:{self.server_id}] Registration event sent: {event}")
        except Exception as e:
            logger.error(f"[Producer:{self.server_id}] Registration failed: {e}")

    def _heartbeat_loop(self):
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL_SECONDS)
            if self._running:
                self._register(event="heartbeat")

    @staticmethod
    def _delivery_callback(err, msg):
        if err:
            logger.warning(f"[Producer] Delivery failed: {err}")
