"""
IDSPipeline — Orchestrates all stages for both test and production modes.

Test mode:     CSV → DatasetSlicer → MultiServerReplay → asyncio.Queue → Stages
Production:    Kafka consumer → asyncio.Queue → Stages
"""

import asyncio
import logging
import os
import time
from typing import Optional, List

logger = logging.getLogger(__name__)


class LatencyStats:
    """Collects end-to-end latency samples for the benchmark report."""

    def __init__(self):
        self.samples: List[float] = []
        self._lock = asyncio.Lock()

    async def record_async(self, latency_ms: float):
        async with self._lock:
            self.samples.append(latency_ms)

    def record(self, latency_ms: float):
        self.samples.append(latency_ms)

    def report(self) -> dict:
        import numpy as np
        if not self.samples:
            return {"count": 0}
        a = np.array(self.samples)
        return {
            "count": len(a),
            "mean_ms": float(np.mean(a)),
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "min_ms": float(np.min(a)),
            "max_ms": float(np.max(a)),
            "throughput_per_sec": len(a) / (np.sum(a) / 1000) if np.sum(a) > 0 else 0,
        }


class IDSPipeline:
    """
    Full IDS pipeline orchestrator.

    Responsibilities:
      - Build stage objects (shared detector/council/defender instances)
      - Wire inter-stage asyncio.Queues
      - Launch stages as concurrent asyncio Tasks
      - Feed data (test: from CSV replay, production: from Kafka consumer)
      - Collect latency stats
      - Expose live_stats() for the Server Manager UI
    """

    QUEUE_MAX_SIZE = 500   # Backpressure buffer per inter-stage queue

    def __init__(self, detector, council, defender, preprocessor, mode: str = "test"):
        self.detector = detector
        self.council = council
        self.defender = defender
        self.preprocessor = preprocessor
        self.mode = mode
        self.stats = LatencyStats()

        # Build inter-stage queues.
        # NOTE: threat_queue was previously capped at 4 which immediately stalled
        # EnsembleDetectionStage whenever the LLM council was busy — every 5th
        # threat would trigger the QueueFull fast-path. Now sized the same as all
        # other queues so the council can drain at its own (slow) pace without
        # creating artificial back-pressure on the hot detection path.
        self.source_queue: asyncio.Queue = asyncio.Queue(maxsize=self.QUEUE_MAX_SIZE)
        self.detection_queue: asyncio.Queue = asyncio.Queue(maxsize=self.QUEUE_MAX_SIZE)
        self.threat_queue: asyncio.Queue = asyncio.Queue(maxsize=self.QUEUE_MAX_SIZE)
        self.benign_queue: asyncio.Queue = asyncio.Queue(maxsize=self.QUEUE_MAX_SIZE)
        self.defender_queue: asyncio.Queue = asyncio.Queue(maxsize=self.QUEUE_MAX_SIZE)

        self._live_stats = {
            "total_flows": 0,
            "threats": 0,
            "benign": 0,
            "actions": {},
            "servers": {},
        }
        self.kafka_consumer: object = None  # set in run_production
        self._detection_stage = None  # set after stage creation
        self._defender_stage = None   # set after stage creation

    # ─────────────────────────────────────────────────────────
    #  Public run methods
    # ─────────────────────────────────────────────────────────

    async def run_test(
        self,
        data_path: str,
        n_servers: int = 3,
        flows_per_second_per_server: float = 0,   # 0 = unlimited burst
        max_flows_per_server: Optional[int] = None,
    ):
        """
        Test mode: slice the dataset, replay all servers concurrently,
        run the full pipeline, return latency stats.
        """
        from src.simulation.dataset_slicer import DatasetSlicer
        from src.simulation.replay_engine import MultiServerReplay
        from src.pipeline.stages import (
            FlowIngestionStage, EnsembleDetectionStage,
            ThreatCouncilStage, DefenderStage
        )

        logger.info(f"[Pipeline] TEST MODE | {n_servers} virtual servers | {data_path}")

        # Slice dataset
        slicer = DatasetSlicer(data_path, n_servers=n_servers, strategy="stratified")
        slices = slicer.slice()

        # Build stages
        ingestion = FlowIngestionStage(self.preprocessor, self.detection_queue)
        detection = EnsembleDetectionStage(self.detector, self.threat_queue, self.benign_queue)
        council = ThreatCouncilStage(self.council, self.defender_queue)
        defender_stage = DefenderStage(self.defender, stats_collector=self.stats)

        # Build multi-server replay
        replay = MultiServerReplay(slices, flows_per_second_per_server=flows_per_second_per_server)

        t_start = time.perf_counter()

        # Launch all pipeline stages concurrently
        pipeline_tasks = [
            asyncio.create_task(ingestion.run(self.source_queue), name="ingestion"),
            asyncio.create_task(detection.run(self.detection_queue), name="detection"),
            asyncio.create_task(council.run(self.threat_queue), name="council"),
            asyncio.create_task(
                defender_stage.run(self.defender_queue, self.benign_queue), name="defender"
            ),
        ]

        # Replay all servers concurrently into the source queue
        await replay.replay_all_to_queue(self.source_queue, max_flows_per_server=max_flows_per_server)
        # Signal end of stream
        await self.source_queue.put(None)

        # Wait for all stages to finish
        await asyncio.gather(*pipeline_tasks)

        elapsed = time.perf_counter() - t_start
        report = self.stats.report()
        report["total_elapsed_s"] = round(elapsed, 3)
        report["mode"] = "test"
        report["n_servers"] = n_servers

        logger.info(f"\n[Pipeline] TEST COMPLETE in {elapsed:.2f}s")
        logger.info(f"           Flows processed: {report.get('count', 0)}")
        logger.info(f"           Mean latency:    {report.get('mean_ms', 0):.1f}ms")
        logger.info(f"           P99 latency:     {report.get('p99_ms', 0):.1f}ms")
        return report

    async def run_production(self, registry):
        """
        Production mode: read from Kafka consumer → pipeline stages.
        Runs indefinitely until KeyboardInterrupt.
        """
        from src.pipeline.stages import (
            FlowIngestionStage, EnsembleDetectionStage,
            ThreatCouncilStage, DefenderStage
        )
        from src.kafka.consumer import FlowConsumer

        logger.info("[Pipeline] PRODUCTION MODE | Waiting for Kafka flows...")

        main_loop = asyncio.get_running_loop()

        def on_kafka_message(raw_flow_msg):
            """
            Callback from Kafka consumer thread → put into asyncio queue.

            FIX: The original `future.result()` (no timeout) blocked the Kafka
            consumer thread indefinitely whenever the source_queue was full. This
            caused `max.poll.interval.ms` (300 s) violations and consumer group
            rebalances, making the pipeline look "stuck" between server switches.

            We now use a 5-second timeout. If the queue is still full after 5 s
            the message is dropped and a warning is logged — far better than a
            silent Kafka rebalance that kills all consumers.
            """
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.source_queue.put(raw_flow_msg), main_loop
                )
                future.result(timeout=5.0)  # 5 s max; raises TimeoutError if queue is backed up
            except TimeoutError:
                logger.warning(
                    "[Pipeline] source_queue full for >5 s — dropping flow to prevent "
                    "Kafka consumer rebalance. Consider raising QUEUE_MAX_SIZE or "
                    "slowing the producer."
                )
            except Exception as e:
                logger.error(f"[Pipeline] Failed to enqueue Kafka message: {e}")

        self.kafka_consumer = FlowConsumer(callback=on_kafka_message, registry=registry)
        self.kafka_consumer.start()

        # Build stages
        ingestion = FlowIngestionStage(self.preprocessor, self.detection_queue)
        detection = EnsembleDetectionStage(self.detector, self.threat_queue, self.benign_queue)
        council_stage = ThreatCouncilStage(self.council, self.defender_queue)
        defender_stage = DefenderStage(self.defender, stats_collector=self.stats)
        # expose for live_stats
        self._detection_stage = detection
        self._defender_stage = defender_stage

        try:
            await asyncio.gather(
                ingestion.run(self.source_queue),
                detection.run(self.detection_queue),
                council_stage.run(self.threat_queue),
                defender_stage.run(self.defender_queue, self.benign_queue),
            )
        except asyncio.CancelledError:
            logger.info("[Pipeline] Production pipeline stopped.")
        finally:
            self.kafka_consumer.stop()

    def live_stats(self) -> dict:
        """Return current pipeline stats for the Server Manager UI."""
        # Sync real counts from stage objects whenever they exist
        if self._detection_stage:
            self._live_stats["threats"] = self._detection_stage._threats_detected
            self._live_stats["benign"] = self._detection_stage._benign_count
        if self._defender_stage:
            self._live_stats["actions_taken"] = self._defender_stage._actions_taken
        return {
            **self._live_stats,
            "latency": self.stats.report(),
        }
