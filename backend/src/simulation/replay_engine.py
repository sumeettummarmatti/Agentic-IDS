"""
Replay Engine — replays a ServerSlice through a FlowProducer at a
configurable rate, mimicking a real geo-server sending live traffic to Kafka.

Each row in the slice is converted to a RawFlowMessage and published.
The engine supports:
  - Rate limiting (flows_per_second)
  - Burst mode (no rate limit — max throughput test)
  - Test mode (no Kafka — puts directly into an asyncio.Queue)
"""

import asyncio
import logging
import time
import threading
from typing import Optional, Callable

import pandas as pd

from src.simulation.dataset_slicer import ServerSlice
from src.kafka.producer import FlowProducer
from src.kafka.registry import ServerRegistry

logger = logging.getLogger(__name__)


class ReplayEngine:
    """
    Replays one ServerSlice at a controlled rate.

    Usage (production with Kafka):
        engine = ReplayEngine(server_slice, flows_per_second=50)
        engine.start_kafka_replay(bootstrap_servers="localhost:9092")

    Usage (test mode — no Kafka):
        queue = asyncio.Queue()
        engine = ReplayEngine(server_slice, flows_per_second=0)   # 0 = burst
        await engine.replay_to_queue(queue)
    """

    def __init__(
        self,
        server_slice: ServerSlice,
        flows_per_second: float = 100.0,   # 0 = unlimited burst
        loop: bool = False,                # Loop the dataset indefinitely
    ):
        self.slice = server_slice
        self.flows_per_second = flows_per_second
        self.loop = loop
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._published = 0
        self._start_time: Optional[float] = None

    # ─────────────────────────────────────────────────────────
    #  Kafka Replay (Production / Simulation Mode)
    # ─────────────────────────────────────────────────────────

    def start_kafka_replay(self, bootstrap_servers: str = "localhost:9092", on_done: Optional[Callable] = None):
        """
        Start replaying this slice to Kafka in a background thread.
        Non-blocking — returns immediately.
        """
        producer = FlowProducer(
            server_id=self.slice.server_id,
            geo_region=self.slice.geo_region,
            geo_label=self.slice.geo_label,
            lat=self.slice.lat,
            lon=self.slice.lon,
            ip="simulator",
            interface="simulated",
            bootstrap_servers=bootstrap_servers,
        )
        producer.start()

        self._running = True
        self._start_time = time.perf_counter()

        def _run():
            try:
                df = self.slice.df
                iteration = 0
                while self._running and (self.loop or iteration == 0):
                    for idx, row in df.iterrows():
                        if not self._running:
                            break
                        features = row.to_dict()
                        flow_id = f"{self.slice.server_id}-{iteration}-{idx}"
                        producer.publish_flow(features, flow_id=flow_id)
                        self._published += 1

                        # Rate limiting
                        if self.flows_per_second > 0:
                            time.sleep(1.0 / self.flows_per_second)

                    iteration += 1

                logger.info(
                    f"[Replay:{self.slice.server_id}] Done. "
                    f"{self._published} flows in {time.perf_counter() - self._start_time:.2f}s"
                )
            finally:
                producer.stop()
                if on_done:
                    on_done(self.slice.server_id, self._published)

        self._thread = threading.Thread(target=_run, daemon=True, name=f"replay-{self.slice.server_id}")
        self._thread.start()

    def stop(self):
        """Stop the replay loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ─────────────────────────────────────────────────────────
    #  Test Mode — direct asyncio.Queue (no Kafka)
    # ─────────────────────────────────────────────────────────

    async def replay_to_queue(self, queue: asyncio.Queue, max_flows: Optional[int] = None):
        """
        Replay the slice directly into an asyncio.Queue (bypasses Kafka).
        Used in --mode test to avoid needing a running Kafka broker.
        """
        from src.kafka.schemas import RawFlowMessage

        df = self.slice.df
        count = 0
        self._start_time = time.perf_counter()

        for idx, row in df.iterrows():
            if max_flows and count >= max_flows:
                break

            features = {
                k: (float(v) if hasattr(v, 'item') else v)
                for k, v in row.to_dict().items()
                if k != "Label"   # Don't leak labels into the pipeline
            }
            msg = RawFlowMessage(
                server_id=self.slice.server_id,
                geo_region=self.slice.geo_region,
                flow_id=f"{self.slice.server_id}-{idx}",
                features=features,
            )
            await queue.put(msg)
            count += 1

            # Rate limiting (0 = async burst)
            if self.flows_per_second > 0:
                await asyncio.sleep(1.0 / self.flows_per_second)

        elapsed = time.perf_counter() - self._start_time
        actual_rate = count / elapsed if elapsed > 0 else float("inf")
        logger.info(
            f"[Replay:{self.slice.server_id}] Queued {count} flows "
            f"in {elapsed:.2f}s ({actual_rate:.0f} flows/s)"
        )
        return count, elapsed


class MultiServerReplay:
    """
    Orchestrates multiple ReplayEngines concurrently.
    Each server replays in parallel (thread per server for Kafka mode,
    concurrent coroutines for test mode).
    """

    def __init__(self, slices: list, flows_per_second_per_server: float = 50.0):
        self.engines = [
            ReplayEngine(s, flows_per_second=flows_per_second_per_server)
            for s in slices
        ]

    def start_all_kafka(self, bootstrap_servers: str = "localhost:9092"):
        """Start all server replays in parallel (Kafka mode)."""
        for engine in self.engines:
            engine.start_kafka_replay(bootstrap_servers=bootstrap_servers)
        logger.info(f"[MultiReplay] Started {len(self.engines)} server replay engines")

    async def replay_all_to_queue(self, queue: asyncio.Queue, max_flows_per_server: Optional[int] = None):
        """
        Replay all servers concurrently into a shared asyncio.Queue (test mode).
        All coroutines run concurrently — mimics parallel geo-server traffic.
        """
        tasks = [
            engine.replay_to_queue(queue, max_flows=max_flows_per_server)
            for engine in self.engines
        ]
        results = await asyncio.gather(*tasks)
        total = sum(r[0] for r in results)
        logger.info(f"[MultiReplay] All servers done. Total flows queued: {total}")
        return total

    def stop_all(self):
        for engine in self.engines:
            engine.stop()
