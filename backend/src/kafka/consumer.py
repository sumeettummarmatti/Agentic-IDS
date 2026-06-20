"""
Kafka consumers for Agentic-IDS.

FlowConsumer           — reads 'raw-flows' topic for the IDS pipeline
ServerRegistryConsumer — reads 'server-registry' and updates the live ServerRegistry
"""

import json
import logging
import os
import threading
import time
from typing import Callable, List, Optional

from .schemas import RawFlowMessage, ServerRegistration
from .registry import ServerRegistry

logger = logging.getLogger(__name__)


class FlowConsumer:
    """
    Consumes 'raw-flows' topic for the IDS detection pipeline.

    Kafka partition pause/resume is used to control which geo-servers
    are actively analysed. All partitions start PAUSED — the UI calls
    monitor(server_id) / unmonitor(server_id) to activate a region.
    Messages backlog safely in Kafka while a partition is paused.

    Usage:
        consumer = FlowConsumer(callback=my_fn, registry=registry)
        consumer.start()
        consumer.monitor("MUM-01")    # start analysing Mumbai
        consumer.unmonitor("MUM-01")  # pause again
        consumer.stop()
    """

    def __init__(
        self,
        callback: Callable[[RawFlowMessage], None],
        group_id: Optional[str] = None,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        auto_offset_reset: str = "latest",
        registry: Optional[ServerRegistry] = None,
    ):
        self.callback = callback
        import uuid
        self.group_id = group_id or f"agentic-ids-group-{uuid.uuid4().hex[:8]}"
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.topic = topic or os.getenv("KAFKA_RAW_FLOWS_TOPIC", "raw-flows")
        self.auto_offset_reset = auto_offset_reset
        self.registry = registry

        self._consumer = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._consumed_count = 0

        # Set of server_ids whose partition is currently ACTIVE (not paused)
        self._monitored: set = set()
        self._lock = threading.Lock()

        # server_id → assigned TopicPartition (populated after assignment)
        self._partition_map: dict = {}

    # ─────────────────────────────────────────────────────────
    #  Public monitor / unmonitor API
    # ─────────────────────────────────────────────────────────

    def monitor(self, server_id: str) -> bool:
        """Resume the Kafka partition for server_id. Returns True if successful."""
        with self._lock:
            self._monitored.add(server_id)
        self._apply_partition_state()
        logger.info(f"[Consumer] ▶ Monitoring STARTED for {server_id}")
        return True

    def unmonitor(self, server_id: str) -> bool:
        """Pause the Kafka partition for server_id. Returns True if successful."""
        with self._lock:
            self._monitored.discard(server_id)
        self._apply_partition_state()
        logger.info(f"[Consumer] ⏸ Monitoring PAUSED for {server_id}")
        return True

    def monitored_servers(self) -> list:
        with self._lock:
            return list(self._monitored)

    def _apply_partition_state(self):
        """
        We no longer pause/resume Kafka partitions.
        Since partitions are hashed by server_id, our UI partition map
        does not align perfectly with Kafka's physical partitions.
        Instead, we consume all partitions and simply drop unmonitored
        flows instantly at the application layer.
        """
        pass

    def _on_assign(self, consumer, partitions):
        """Callback when Kafka assigns partitions to this consumer."""
        logger.info(f"[Consumer] Partition assignment: {[p.partition for p in partitions]}")

        # Build a server_id → TopicPartition map using the registry
        with self._lock:
            self._partition_map.clear()
            if self.registry:
                for server in self.registry.get_all():
                    for p in partitions:
                        if p.partition == server.kafka_partition:
                            self._partition_map[server.server_id] = p
                            break
            # Any unmatched partition → placeholder so we can track it
            for p in partitions:
                if not any(tp.partition == p.partition for tp in self._partition_map.values()):
                    self._partition_map[f"__unknown_p{p.partition}"] = p

        # NOTE: We do NOT pause partitions here because the registry is often
        # empty at assignment time (Docker heartbeats arrive later).
        # Filtering is done by the _monitored set check in _consume_loop.
        logger.info("[Consumer] All partitions active — activate a region in the UI to start analysis")

    # ─────────────────────────────────────────────────────────
    #  Lifecycle
    # ─────────────────────────────────────────────────────────

    def start(self):
        """Connect to Kafka and start consuming in a background thread."""
        try:
            from confluent_kafka import Consumer, KafkaException

            self._consumer = Consumer({
                "bootstrap.servers": self.bootstrap_servers,
                "group.id": self.group_id,
                "auto.offset.reset": self.auto_offset_reset,
                "enable.auto.commit": False,
                "max.poll.interval.ms": 300000,
                "session.timeout.ms": 45000,
                # fetch.wait.max.ms was 500 — this added up to 500 ms of forced
                # latency per fetch RPC when the topic was quiet. 50 ms is a
                # much better tradeoff between broker round-trips and latency.
                "fetch.wait.max.ms": 50,
                # Require at least 1 KB before the broker replies so it can
                # aggregate a micro-batch and reduce per-fetch overhead.
                "fetch.min.bytes": 1024,
                # Cap in-flight message buffer to avoid unbounded memory growth
                # when the pipeline is slower than the producer.
                "queued.max.messages.kbytes": 32768,  # 32 MB
            })
            self._consumer.subscribe([self.topic], on_assign=self._on_assign)
            logger.info(
                f"[Consumer] Subscribed to '{self.topic}' "
                f"(group={self.group_id}, offset={self.auto_offset_reset})"
            )
        except ImportError:
            logger.warning("[Consumer] confluent_kafka not installed — will use no-op mode")
            self._consumer = None

        self._running = True
        self._thread = threading.Thread(
            target=self._consume_loop, daemon=True, name="flow-consumer"
        )
        self._thread.start()

    def stop(self):
        """Gracefully stop the consumer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        if self._consumer:
            self._consumer.close()
        logger.info(f"[Consumer] Stopped. Total consumed: {self._consumed_count}")

    def commit(self):
        """Manually commit current offset."""
        if self._consumer:
            try:
                self._consumer.commit(asynchronous=True)
            except Exception as e:
                logger.warning(f"[Consumer] Commit failed: {e}")

    def _consume_loop(self):
        if not self._consumer:
            logger.warning("[Consumer] No Kafka connection — consumer loop exiting")
            return

        from confluent_kafka import KafkaException

        while self._running:
            try:
                # Drain up to 100 messages per iteration to reduce per-message
                # overhead (poll() has a fixed cost regardless of batch size).
                msgs = self._consumer.consume(num_messages=100, timeout=0.05)
                if not msgs:
                    continue

                for msg in msgs:
                    if msg is None:
                        continue
                    if msg.error():
                        logger.error(f"[Consumer] Kafka error: {msg.error()}")
                        continue

                    raw = RawFlowMessage.from_json(msg.value().decode("utf-8"))

                    # Only forward if the source server is currently monitored
                    with self._lock:
                        active = raw.server_id in self._monitored
                    if active:
                        self.callback(raw)
                        self._consumed_count += 1

                # Commit periodically (every ~500 messages)
                if self._consumed_count > 0 and self._consumed_count % 500 == 0:
                    self.commit()

            except KafkaException as e:
                logger.error(f"[Consumer] KafkaException: {e}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"[Consumer] Unexpected error: {e}")
                time.sleep(1)


class ServerRegistryConsumer:
    """
    Subscribes to 'server-registry' topic and keeps the local ServerRegistry in sync.

    On startup, replays from 'earliest' to reconstruct registry state before
    any new events arrive (compacted topic ensures we get the latest per server).
    """

    def __init__(
        self,
        registry: ServerRegistry,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
    ):
        self.registry = registry
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.topic = topic or os.getenv("KAFKA_SERVER_REGISTRY_TOPIC", "server-registry")
        self._consumer = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        try:
            from confluent_kafka import Consumer

            self._consumer = Consumer({
                "bootstrap.servers": self.bootstrap_servers,
                # Unique group each run → always replays from earliest
                "group.id": f"agentic-ids-registry-{int(time.time())}",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
            })
            self._consumer.subscribe([self.topic])
            logger.info(f"[RegistryConsumer] Subscribed to '{self.topic}' (replaying from earliest)")
        except ImportError:
            logger.warning("[RegistryConsumer] confluent_kafka not installed — skipping")
            self._consumer = None
            return

        self._thread = threading.Thread(
            target=self._consume_loop, daemon=True, name="registry-consumer"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._consumer:
            self._consumer.close()

    def _consume_loop(self):
        if not self._consumer:
            return

        while self._running:
            try:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue

                reg = ServerRegistration.from_json(msg.value().decode("utf-8"))

                if reg.event == "register" or reg.event == "heartbeat":
                    self.registry.register(reg)
                    if reg.event == "heartbeat":
                        self.registry.heartbeat(reg.server_id)
                elif reg.event == "deregister":
                    self.registry.deregister(reg.server_id)

            except Exception as e:
                logger.error(f"[RegistryConsumer] Error: {e}")
                time.sleep(1)
