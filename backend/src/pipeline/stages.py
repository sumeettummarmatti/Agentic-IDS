"""
Async IDS Pipeline — 4 stage nodes connected by asyncio.Queues.

Stage 1: FlowIngestionStage   — reads RawFlowMessage, bridges features, preprocesses
Stage 2: EnsembleDetectionStage — XGBoost + LSTM scoring, routes by confidence
Stage 3: ThreatCouncilStage   — LLM council + Karpathy (for high-confidence threats)
Stage 4: DefenderStage        — PPO RL agent → action → Kafka decisions topic

All stages run as asyncio.Tasks. Inter-stage communication = asyncio.Queue.
The shared model objects (EnsembleDetector, DefenderRLAgent) are loaded once
and used concurrently in read-only inference mode.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

from src.agents.kernel_executor import KernelExecutor

logger = logging.getLogger(__name__)


def _write_council_patch(server_id: str, flow_id: str, council_result: dict, rl_action: str = None, rl_action_id: int = None, commands: list = None, command_comments: list = None):
    """
    Append a council_update record to the per-server JSONL file.

    When the defender writes the initial report, council enrichment hasn't
    finished yet so severity/council fields are 'pending'.
    This patch record lets the frontend update the existing row in-place
    by matching flow_id.
    """
    import json as _json
    from pathlib import Path
    from datetime import datetime, timezone

    reports_dir = Path("reports/actions")
    reports_dir.mkdir(parents=True, exist_ok=True)
    fpath = reports_dir / f"{server_id}_actions.jsonl"

    patch = {
        "_type": "council_update",
        "flow_id": flow_id,
        "server_id": server_id,
        "council_severity":       council_result.get("severity", "Unknown"),
        "council_recommendation": council_result.get("recommendations", []),
        "council_consensus":      council_result.get("council_consensus", 0.0),
        "council_threat_type":    council_result.get("threat_type", ""),
        "council_signature":      council_result.get("signature_match", ""),
        "council_threat_actor":   council_result.get("threat_actor_type", ""),
        "council_fp_risk":        council_result.get("false_positive_risk", ""),
        "agent_votes":            council_result.get("agent_votes", {}),
        "timestamp_utc":          datetime.now(timezone.utc).isoformat(),
        "raw_responses":          council_result.get("raw_responses", {}),
    }
    
    if rl_action is not None:
        patch["final_rl_action"] = rl_action
    if rl_action_id is not None:
        patch["rl_action_id"] = rl_action_id
    if commands is not None:
        patch["commands"] = commands
    if command_comments is not None:
        patch["command_comments"] = command_comments

    try:
        with open(fpath, "a") as f:
            f.write(_json.dumps(patch) + "\n")
    except Exception as e:
        logger.error(f"[Council] Failed to write council patch for {flow_id}: {e}")

# Lazy import so stages.py doesn't hard-depend on the UI module.
# emit_pipeline_event is a no-op if server_manager isn't loaded (test mode).
def _emit(event: dict):
    try:
        from src.ui import server_manager as sm
        sm.emit_pipeline_event(event)
    except Exception:
        pass  # Not running in UI mode

# Internal sentinels
_STOP_SENTINEL = None

# ── Ingestion tuning ─────────────────────────────────────────
# Drain up to this many messages per iteration before handing off to detection.
# Larger = better GPU/CPU batching; smaller = lower latency. 64 is a good default.
_INGESTION_BATCH_SIZE = 64

# ── Detection tuning ─────────────────────────────────────────
# Maximum number of flows batched together for XGBoost / LSTM inference.
_DETECTION_BATCH_SIZE = 64

# Timeout (seconds) to flush a partial batch even when the queue is quiet.
_BATCH_DRAIN_TIMEOUT = 0.05   # 50 ms — keeps latency bounded


# ─────────────────────────────────────────────────────────────
#  Stage 1: Flow Ingestion
# ─────────────────────────────────────────────────────────────

class FlowIngestionStage:
    """
    Reads RawFlowMessage objects from the source queue in micro-batches,
    applies vectorised numeric conversion + zero-fill, runs the Preprocessor
    transform on the whole batch, and pushes enriched items to the detection
    queue.

    FIX 1: Replaced per-row `pd.DataFrame([msg.features])` + `apply(pd.to_numeric)`
    (extremely expensive — rebuilds a new DataFrame object per flow) with a
    batch approach that builds one DataFrame per N flows.

    FIX 2: Replaced `__import__("pandas").to_numeric` hack (re-imports pandas
    every single call) with a direct `pd.to_numeric` reference captured at
    module import time.
    """

    def __init__(self, preprocessor, detection_queue: asyncio.Queue):
        self.preprocessor = preprocessor
        self.detection_queue = detection_queue
        self._processed = 0

    async def run(self, source_queue: asyncio.Queue):
        """Consume from source_queue until STOP_SENTINEL."""
        import pandas as pd

        logger.info("[Ingestion] Stage started")
        stop_received = False

        while not stop_received:
            # ── Drain a batch ────────────────────────────────────────
            batch: List = []
            try:
                # Block on first item to avoid a busy-wait spin
                first = await source_queue.get()
                if first is _STOP_SENTINEL:
                    stop_received = True
                else:
                    batch.append(first)

                # Non-blocking drain for the rest of the batch
                while len(batch) < _INGESTION_BATCH_SIZE:
                    try:
                        item = source_queue.get_nowait()
                        if item is _STOP_SENTINEL:
                            stop_received = True
                            break
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break
            except Exception as e:
                logger.error(f"[Ingestion] Queue read error: {e}")
                break

            if not batch:
                continue

            # ── Process entire batch at once ─────────────────────────
            try:
                t_start = time.perf_counter()

                # Build one DataFrame for the whole batch — far cheaper than N DataFrames
                # Strip leading/trailing spaces from feature names to match preprocessor
                rows = [
                    {k.strip(): v for k, v in msg.features.items()}
                    for msg in batch
                ]
                df = pd.DataFrame(rows)
                df = df.apply(pd.to_numeric, errors="coerce")
                df = df.fillna(0.0)

                # Run preprocessor on the batch
                if self.preprocessor.feature_names:
                    X, _, _ = self.preprocessor.prepare_features_and_labels(df, training=False)
                else:
                    X = df.values

                ingestion_latency = (time.perf_counter() - t_start) * 1000  # ms

                # Push each enriched item downstream
                for i, msg in enumerate(batch):
                    row_X = X[i] if X.ndim > 1 else X
                    await self.detection_queue.put({
                        "msg": msg,
                        "X": row_X,
                        "t_ingested": time.perf_counter(),
                        "ingestion_latency_ms": ingestion_latency / len(batch),
                    })

                self._processed += len(batch)

            except Exception as e:
                logger.error(f"[Ingestion] Error processing batch of {len(batch)}: {e}")

        await self.detection_queue.put(_STOP_SENTINEL)
        logger.info(f"[Ingestion] Done. Processed: {self._processed}")


# ─────────────────────────────────────────────────────────────
#  Stage 2: Ensemble Detection
# ─────────────────────────────────────────────────────────────

class EnsembleDetectionStage:
    """
    Runs XGBoost + LSTM ensemble on batches of flows for maximum throughput.
    Routes high-confidence threats to the council queue.
    Routes benign flows to the benign queue (audit only).

    FIX 3: Replaced per-flow `predict_proba(reshape(1,-1))` with batched
    `predict_proba(X_batch)` — XGBoost's predict_proba has significant per-call
    overhead (GIL acquisition, validation, output allocation). Batching 64 flows
    at once is ~10-20× faster than 64 individual calls.

    FIX 4: Replaced per-flow LSTM forward pass with a batched tensor pass.
    Running N flows as a single (N, 1, features) tensor uses the full BLAS/cuBLAS
    pipeline instead of looping over (1, 1, features) scalars.

    FIX 5: `asyncio.gather` on the drain loop with a timeout prevents the stage
    from blocking forever when flows arrive slowly.
    """

    CONFIDENCE_THRESHOLD = 0.6
    CLASS_NAMES = {0: "BENIGN", 1: "DDoS", 2: "PortScan"}

    def __init__(self, detector, threat_queue: asyncio.Queue, benign_queue: asyncio.Queue):
        self.detector = detector
        self.threat_queue = threat_queue
        self.benign_queue = benign_queue
        self._threats_detected = 0
        self._benign_count = 0

    async def run(self, detection_queue: asyncio.Queue):
        import torch

        logger.info("[Detection] Stage started")
        stop_received = False

        while not stop_received:
            # ── Drain a batch (with timeout to flush partial batches) ─
            batch: List[Dict] = []
            deadline = time.monotonic() + _BATCH_DRAIN_TIMEOUT

            try:
                first = await detection_queue.get()
                if first is _STOP_SENTINEL:
                    stop_received = True
                else:
                    batch.append(first)

                while len(batch) < _DETECTION_BATCH_SIZE:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = detection_queue.get_nowait()
                        if item is _STOP_SENTINEL:
                            stop_received = True
                            break
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        # Nothing ready yet — yield briefly and retry
                        await asyncio.sleep(0)
                        continue
            except Exception as e:
                logger.error(f"[Detection] Queue read error: {e}")
                break

            if not batch:
                continue

            # ── Batch inference ──────────────────────────────────────
            try:
                # Stack all feature vectors → (N, features)
                X_batch = np.stack([item["X"] for item in batch])

                # XGBoost — single batched call
                xgb_probs = self.detector.xgb_model.predict_proba(X_batch)  # (N, C_xgb)

                # LSTM — single batched forward pass
                features_tensor = torch.FloatTensor(X_batch).unsqueeze(1)  # (N, 1, features)
                with torch.no_grad():
                    lstm_out = self.detector.lstm_model(features_tensor)
                    lstm_probs = torch.softmax(lstm_out, dim=1).numpy()  # (N, C_lstm)

                # Ensemble: only combine when class counts match.
                # If LSTM was initialized with wrong num_classes (e.g. random
                # weights from the old 10-class traffic model) padding XGBoost
                # probs to 10 classes and mixing in uniform LSTM noise would
                # dilute every DDoS/PortScan confidence below the 0.6 threshold,
                # causing all threats to be silently routed to the benign queue.
                if xgb_probs.shape[1] == lstm_probs.shape[1]:
                    final_probs = (0.7 * xgb_probs) + (0.3 * lstm_probs)  # (N, C)
                else:
                    logger.warning(
                        f"[Detection] LSTM class count ({lstm_probs.shape[1]}) != "
                        f"XGBoost ({xgb_probs.shape[1]}) — using XGBoost only"
                    )
                    final_probs = xgb_probs  # (N, C_xgb) — no noise dilution

                pred_indices = np.argmax(final_probs, axis=1)          # (N,)
                confidences = final_probs[np.arange(len(batch)), pred_indices]

                # Map indices to class names
                if hasattr(self.detector, "training_classes_"):
                    classes = self.detector.training_classes_
                    attack_types = []
                    for idx in pred_indices:
                        try:
                            attack_types.append(self.CLASS_NAMES.get(int(classes[idx]), str(idx)))
                        except Exception:
                            attack_types.append(str(idx))
                else:
                    attack_types = [self.CLASS_NAMES.get(int(i), str(i)) for i in pred_indices]

                # Route results
                t_detected = time.perf_counter()
                for item, attack_type, confidence in zip(batch, attack_types, confidences):
                    detection_result = {
                        **item,
                        "attack_type": attack_type,
                        "confidence": float(confidence),
                        "t_detected": t_detected,
                    }

                    if attack_type != "BENIGN" and confidence > self.CONFIDENCE_THRESHOLD:
                        self._threats_detected += 1
                        try:
                            self.threat_queue.put_nowait(detection_result)
                        except asyncio.QueueFull:
                            # Council overwhelmed — fast-path directly to Defender
                            logger.warning(
                                f"[Detection] Council overwhelmed! Fast-pathing {attack_type} "
                                f"to Defender for immediate mitigation."
                            )
                            await self.benign_queue.put(detection_result)
                    else:
                        self._benign_count += 1
                        await self.benign_queue.put(detection_result)

            except Exception as e:
                logger.error(f"[Detection] Error on batch of {len(batch)}: {e}")

        await self.threat_queue.put(_STOP_SENTINEL)
        await self.benign_queue.put(_STOP_SENTINEL)
        logger.info(
            f"[Detection] Done. Threats: {self._threats_detected}, "
            f"Benign: {self._benign_count}"
        )


# ─────────────────────────────────────────────────────────────
#  Stage 3: Threat Council
# ─────────────────────────────────────────────────────────────

class ThreatCouncilStage:
    """
    LLM Council enrichment for high-confidence threats.

    Architecture fix: threats are forwarded to the Defender IMMEDIATELY on
    arrival, so the RL agent can act without waiting for the (slow) LLM.
    Council analysis then runs in the background (rate-limited by semaphore)
    and logs its recommendation — it is enrichment, not a blocker.

    Root cause of the original bug:
      - Council held every threat until the LLM returned (2-15 s with Ollama).
      - During that time the defender_queue was empty, so DefenderStage only
        consumed from benign_queue — producing the symptom of "898 threats
        detected but zero threat logs in the defender".
    """

    def __init__(self, council, defender_queue: asyncio.Queue, decisions_producer=None):
        self.council = council
        self.defender_queue = defender_queue
        self.decisions_producer = decisions_producer
        self._analyzed = 0
        # Rate-limit LLM calls so they don't monopolise the thread-pool executor.
        self._semaphore = asyncio.Semaphore(5)  # serialize LLM calls — free-tier Groq has tight TPM limits

        self._pending_tasks: set = set()

    async def run(self, threat_queue: asyncio.Queue):
        logger.info("[Council] Stage started")
        while True:
            item = await threat_queue.get()
            if item is _STOP_SENTINEL:
                # Drain any in-flight enrichment tasks before shutting down
                if self._pending_tasks:
                    await asyncio.gather(*self._pending_tasks, return_exceptions=True)
                await self.defender_queue.put(_STOP_SENTINEL)
                logger.info(f"[Council] Done. Analyzed: {self._analyzed}")
                break

            # Emit an immediate "queued" event so the UI sees the threat arrive
            msg = item.get("msg")
            if msg:
                _emit({
                    "stage": "council",
                    "event": "queued",
                    "flow_id": msg.flow_id,
                    "server_id": msg.server_id,
                    "attack_type": item.get("attack_type"),
                    "confidence": round(item.get("confidence", 0) * 100, 1),
                })

            task = asyncio.create_task(self._process_threat(item))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

    async def _process_threat(self, item):
        msg = item["msg"]
        attack_type = item["attack_type"]
        confidence = item["confidence"]

        # ── Step 1: Forward threat to Defender IMMEDIATELY ──────────
        await self.defender_queue.put({
            **item,
            "council_result": None,
            "t_council_done": time.perf_counter(),
        })

        # ── Step 2: LLM enrichment in background (rate-limited) ─────
        async with self._semaphore:
            try:
                logger.info(
                    f"[Council] Enriching: {msg.flow_id} | "
                    f"{attack_type} ({confidence:.1%}) from {msg.server_id}"
                )
                _emit({
                    "stage": "council",
                    "event": "enriching",
                    "flow_id": msg.flow_id,
                    "server_id": msg.server_id,
                    "attack_type": attack_type,
                    "confidence": round(confidence * 100, 1),
                })
                flow_dict = msg.features
                prediction = {"attack_type": attack_type, "confidence": confidence}

                loop = asyncio.get_running_loop()
                council_result = await loop.run_in_executor(
                    None,
                    lambda: self.council.analyze_threat(flow_dict, prediction)
                )

                self._analyzed += 1
                recommended = (
                    council_result.get("recommended_action", "?") if council_result else "?"
                )
                logger.info(
                    f"[Council] ✓ {msg.flow_id} enriched — "
                    f"recommendation: {recommended}"
                )

                # Emit full council result so pipeline tab can display it
                if council_result:
                    _emit({
                        "stage": "council",
                        "event": "complete",
                        "flow_id": msg.flow_id,
                        "server_id": msg.server_id,
                        "attack_type": council_result.get("threat_type", attack_type),
                        "severity": council_result.get("severity", "Medium"),
                        "recommended_action": recommended,
                        "confidence": round(council_result.get("confidence", confidence) * 100, 1),
                        "agent_votes": council_result.get("agent_votes", {}),
                        "signature": council_result.get("signature_match", ""),
                        "consensus": round(council_result.get("council_consensus", 0) * 100, 0),
                        "analyst_action": council_result.get("agent_votes", {}).get("analyst", "?"),
                        "engineer_action": council_result.get("agent_votes", {}).get("engineer", "?"),
                        "intel_action": council_result.get("agent_votes", {}).get("intel", "?"),
                    })

                    # ── Write council patch to JSONL so Threat Reports table updates ──
                    # The defender already wrote an initial record with severity=pending.
                    # We forward it again so the RL agent can re-evaluate with the
                    # LLM's highly accurate consensus, and write the council_update patch.
                    await self.defender_queue.put({
                        **item,
                        "council_result": council_result,
                        "is_re_evaluation": True,
                        "t_council_done": time.perf_counter(),
                    })

            except Exception as e:
                logger.error(f"[Council] LLM enrichment error for {msg.flow_id}: {e}")
                _emit({
                    "stage": "council",
                    "event": "error",
                    "flow_id": msg.flow_id,
                    "server_id": msg.server_id,
                    "error": str(e)[:120],
                })


# ─────────────────────────────────────────────────────────────
#  Stage 4: Defender
# ─────────────────────────────────────────────────────────────

class DefenderStage:
    """
    PPO RL agent selects mitigation action.
    Publishes DefenseDecisionMessage to Kafka decisions topic.
    Tracks end-to-end latency per flow.

    FIX 7 (original): asyncio.sleep(0.01) polling loop replaced.

    FIX 8 (this change): asyncio.wait + cancel had a silent item-loss bug.
    When asyncio puts an item into a queue, it calls future.set_result(item).
    If we then cancel that future before the coroutine reads it, the item is
    gone — no exception, no retry. This caused threats from defender_queue
    to silently disappear whenever benign_queue fired first (which is almost
    always, since benign flows far outnumber LLM-gated threats).

    Fix: two lightweight feeder tasks drain each queue into a single merged
    asyncio.Queue. The consumer loop awaits one queue and never cancels a
    live getter, so no item can be lost.
    """

    ACTIONS = {0: "MONITOR", 1: "BLOCK_SOURCE", 2: "DEEP_PACKET_INSPECTION", 3: "RATE_LIMIT"}

    def __init__(self, defender, stats_collector=None, kafka_producer=None,
                 report_dir: str = "reports/actions"):
        self.defender = defender
        self.stats_collector = stats_collector
        self.kafka_producer = kafka_producer
        self._actions_taken = 0
        # Kernel executor — dry_run=True means commands are built + logged but
        # never fired. Flip to False only in a privileged Linux environment.
        self._executor = KernelExecutor(report_dir=report_dir, dry_run=True)

    async def run(self, defender_queue: asyncio.Queue, benign_queue: asyncio.Queue):
        logger.info("[Defender] Stage started")

        # Merged queue — both feeders write here; this loop reads from one place.
        # Unbounded so feeders never block each other (back-pressure is on upstream queues).
        merged: asyncio.Queue = asyncio.Queue()

        async def _drain(q: asyncio.Queue):
            """Forward every item from q into merged, then stop."""
            while True:
                item = await q.get()
                await merged.put(item)
                if item is _STOP_SENTINEL:
                    break

        feeder_tasks = [
            asyncio.create_task(_drain(defender_queue), name="drain-threats"),
            asyncio.create_task(_drain(benign_queue),   name="drain-benign"),
        ]

        pending = 2  # One sentinel expected per feeder
        while pending > 0:
            item = await merged.get()
            if item is _STOP_SENTINEL:
                pending -= 1
                continue
            await self._handle_item(item)

        await asyncio.gather(*feeder_tasks, return_exceptions=True)
        logger.info(f"[Defender] Done. Actions taken: {self._actions_taken}")

    async def _handle_item(self, item):
        msg = item["msg"]
        confidence = item.get("confidence", 0.0)
        attack_type = item.get("attack_type", "BENIGN")
        council_result = item.get("council_result")  # None if LLM still enriching
        t_ingested = item.get("t_ingested", time.perf_counter())
        is_re_evaluation = item.get("is_re_evaluation", False)

        try:
            if attack_type == "BENIGN":
                action = "ALLOW"
                action_result = {"action": "ALLOW", "action_id": -1, "status": "allowed"}
            else:
                if is_re_evaluation and council_result:
                    confidence = council_result.get("confidence", confidence)
                    council_sev = council_result.get("severity", "Medium")
                    threat_level = "High" if council_sev in ["High", "Critical"] else council_sev
                else:
                    threat_level = "High" if confidence > 0.8 else "Medium"
                    
                perception = {
                    "confidence": confidence,
                    "threat_level": threat_level,
                    "flow_rate": float(msg.features.get("Flow Packets/s", 0)),
                }
                obs = self.defender.observe(perception)
                action_result = self.defender.act(obs)
                action = action_result.get("action", "MONITOR")

                # Emit defender decision event for live pipeline view
                _emit({
                    "stage": "defender",
                    "event": "action",
                    "flow_id": msg.flow_id,
                    "server_id": msg.server_id,
                    "attack_type": attack_type,
                    "confidence": round(confidence * 100, 1),
                    "action": action,
                    "latency_ms": round((time.perf_counter() - t_ingested) * 1000, 1),
                })

            end_to_end_ms = (time.perf_counter() - t_ingested) * 1000

            logger.info(
                f"[Defender] {msg.server_id} | {attack_type} ({confidence:.1%}) → "
                f"{action} | latency={end_to_end_ms:.1f}ms"
            )

            # ── Kernel executor: build geo-aware action report ─────────────────
            # Only threat flows get mitigation commands (BENIGN → no kernel action).
            # Runs in a thread executor so file I/O doesn't block the event loop.
            if attack_type != "BENIGN":
                src_ip = msg.features.get("Src IP") or msg.features.get("src_ip")
                src_ip_str = str(src_ip) if src_ip else None
                loop = asyncio.get_running_loop()
                
                if is_re_evaluation:
                    commands, comments = self._executor._build_commands(action, src_ip_str, msg.flow_id)
                    if not self._executor.dry_run and src_ip_str and src_ip_str != "N/A (not in flow features)":
                        await loop.run_in_executor(None, lambda: self._executor._run_commands(commands))
                    
                    await loop.run_in_executor(
                        None,
                        lambda: _write_council_patch(
                            server_id=msg.server_id,
                            flow_id=msg.flow_id,
                            council_result=council_result,
                            rl_action=action,
                            rl_action_id=action_result.get("action_id", 0),
                            commands=commands,
                            command_comments=comments
                        )
                    )
                else:
                    await loop.run_in_executor(
                        None,
                        lambda: self._executor.execute(
                            flow_id=msg.flow_id,
                            server_id=msg.server_id,
                            geo_region=getattr(msg, "geo_region", "unknown"),
                            attack_type=attack_type,
                            confidence=confidence,
                            rl_action=action,
                            rl_action_id=action_result.get("action_id", 0),
                            latency_ms=end_to_end_ms,
                            council_result=council_result,
                            src_ip=src_ip_str,
                        ),
                    )

            # ── Kafka decisions topic ─────────────────────────────────────────
            if self.kafka_producer:
                from src.kafka.schemas import DefenseDecisionMessage
                decision = DefenseDecisionMessage(
                    server_id=msg.server_id,
                    geo_region=getattr(msg, "geo_region", "unknown"),
                    flow_id=msg.flow_id,
                    attack_type=attack_type,
                    confidence=confidence,
                    action=action,
                    action_id=action_result.get("action_id", 0),
                    status=action_result.get("status", "executed"),
                    latency_ms=end_to_end_ms,
                )
                self.kafka_producer.publish_flow(
                    vars(decision), flow_id=msg.flow_id
                )

            if self.stats_collector:
                self.stats_collector.record(end_to_end_ms)

            self._actions_taken += 1

        except Exception as e:
            logger.error(f"[Defender] Error: {e}")
