"""
Server Manager UI — FastAPI app.

Frontend files are served from: <repo-root>/frontend/index.html
Run this app from the backend/ directory:
    cd backend && uvicorn src.ui.server_manager:app --host 0.0.0.0 --port 5001

Endpoints:
  GET  /            → dashboard HTML
  GET  /api/servers → list all registered servers
  POST /api/servers → register a new server (called from UI)
  DELETE /api/servers/{server_id} → deregister
  GET  /api/presets → available geo presets
  GET  /api/stats   → live pipeline stats
  GET  /api/stats/stream → SSE stream for real-time updates

Run:
    uvicorn src.ui.server_manager:app --host 0.0.0.0 --port 5001
"""

import asyncio
import collections
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
    from fastapi.responses import HTMLResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning("FastAPI not installed — Server Manager UI disabled")

from src.kafka.registry import ServerRegistry
from src.kafka.schemas import ServerRegistration

# Global shared state (set by main.py after init)
registry: ServerRegistry = ServerRegistry()
pipeline_stats: dict = {}
active_replays: dict = {}   # server_id → ReplayEngine
pipeline = None             # IDSPipeline instance — set by main.py

# ── Live pipeline event ring-buffer ────────────────────────────────────────
# Stages (council, defender) push dicts here; the /api/pipeline/stream SSE
# endpoint reads them. Thread-safe via a lock on a deque.
_PIPELINE_EVENT_MAXLEN = 500
_pipeline_events: collections.deque = collections.deque(maxlen=_PIPELINE_EVENT_MAXLEN)
_pipeline_events_lock = threading.Lock()
_pipeline_event_counter = 0  # monotonically increasing sequence id


def emit_pipeline_event(event: dict):
    """Called by stages/council to push a live event into the ring-buffer."""
    global _pipeline_event_counter
    with _pipeline_events_lock:
        _pipeline_event_counter += 1
        event["seq"] = _pipeline_event_counter
        event.setdefault("ts", datetime.now(timezone.utc).isoformat())
        _pipeline_events.append(event)


def get_pipeline_events_since(seq: int):
    """Return all events with seq > seq (for SSE polling)."""
    with _pipeline_events_lock:
        return [e for e in _pipeline_events if e.get("seq", 0) > seq]


# ─────────────────────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────────────────────

# Resolve path to the frontend directory (repo-root/frontend/)
# server_manager.py lives at: backend/src/ui/server_manager.py
# repo root = three levels up
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "frontend"))

if HAS_FASTAPI:
    app = FastAPI(title="Agentic-IDS Server Manager", version="1.0")

    # Allow browser to call the API from the same origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static frontend assets
    if os.path.isdir(_FRONTEND_DIR):
        app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="frontend")

    class AddServerRequest(BaseModel):
        server_id: str
        geo_region: str
        geo_label: str
        lat: float
        lon: float
        ip: str = "simulator"
        interface: str = "simulated"
        data_path: Optional[str] = None           # CSV slice to replay
        flows_per_second: float = 50.0
        n_servers_total: int = 1                  # How many total to split data into

    # ── Server CRUD ──────────────────────────────────────────

    @app.get("/api/servers")
    def list_servers():
        return {"servers": registry.to_api_list()}

    @app.get("/api/presets")
    def get_presets():
        return {"presets": ServerRegistry.GEO_PRESETS}

    @app.post("/api/servers", status_code=201)
    def add_server(req: AddServerRequest):
        """
        Register a new geo-server and optionally start replaying data to Kafka.
        If data_path is provided, slices the dataset and starts a ReplayEngine.
        """
        reg = ServerRegistration(
            server_id=req.server_id,
            geo_region=req.geo_region,
            geo_label=req.geo_label,
            lat=req.lat,
            lon=req.lon,
            ip=req.ip,
            interface=req.interface,
            event="register",
        )
        reg = registry.register(reg)
        logger.info(f"[UI] Registered server: {reg.server_id} → partition {reg.kafka_partition}")

        # Optionally start data replay
        if req.data_path and os.path.exists(req.data_path):
            _start_replay(req, reg)

        return {
            "status": "registered",
            "server": {
                "server_id": reg.server_id,
                "kafka_partition": reg.kafka_partition,
                "geo_label": reg.geo_label,
            }
        }

    @app.delete("/api/servers/{server_id}")
    def remove_server(server_id: str):
        if not registry.deregister(server_id):
            raise HTTPException(status_code=404, detail=f"Server {server_id!r} not found")

        # Stop replay if running
        eng = active_replays.pop(server_id, None)
        if eng:
            eng.stop()

        return {"status": "deregistered", "server_id": server_id}

    # ── Monitor / Unmonitor ──────────────────────────────────

    @app.post("/api/servers/{server_id}/monitor", status_code=200)
    def start_monitoring(server_id: str):
        """Activate Kafka partition for a server — start analysing its flows."""
        if pipeline and pipeline.kafka_consumer:
            pipeline.kafka_consumer.monitor(server_id)
            return {"status": "monitoring", "server_id": server_id}
        return {"status": "no_consumer", "server_id": server_id}

    @app.delete("/api/servers/{server_id}/monitor", status_code=200)
    def stop_monitoring(server_id: str):
        """Pause Kafka partition for a server — stop analysing its flows."""
        if pipeline and pipeline.kafka_consumer:
            pipeline.kafka_consumer.unmonitor(server_id)
            return {"status": "paused", "server_id": server_id}
        return {"status": "no_consumer", "server_id": server_id}

    @app.get("/api/monitored")
    def get_monitored():
        """Return the list of server IDs currently being actively monitored."""
        if pipeline and pipeline.kafka_consumer:
            return {"monitored": pipeline.kafka_consumer.monitored_servers()}
        return {"monitored": []}

    # ── Stats & SSE ──────────────────────────────────────────

    @app.get("/api/stats")
    def get_stats():
        lat = pipeline_stats.get("latency", {})
        all_servers = registry.to_api_list()
        return {
            "servers": all_servers,
            "active_servers": len(all_servers),  # all registered, not just heartbeat
            "pipeline": pipeline_stats,
            "total_threats": pipeline_stats.get("threats", 0),
            "total_flows": lat.get("count", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/stats/stream")
    async def stats_stream():
        """
        Server-Sent Events stream — the UI polls this for live updates
        without WebSocket complexity.
        """
        async def event_generator():
            while True:
                lat = pipeline_stats.get("latency", {})
                all_servers = registry.to_api_list()
                data = json.dumps({
                    "servers": all_servers,
                    "active_servers": len(all_servers),  # all registered servers
                    "pipeline": pipeline_stats,
                    "total_threats": pipeline_stats.get("threats", 0),
                    "total_flows": lat.get("count", 0),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                yield f"data: {data}\n\n"
                await asyncio.sleep(2)   # Push every 2 seconds

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/restore-session")
    def restore_session():
        """
        Re-register servers found in reports/actions/*.jsonl into the in-memory
        registry after a server restart. Reads each JSONL file, grabs the first
        record's server metadata, and registers it with a pinned heartbeat so
        the server appears as 'alive' on the dashboard.
        """
        import glob as _glob, json as _json
        from pathlib import Path
        from src.kafka.schemas import ServerRegistration
        from datetime import timedelta

        reports_dir = Path("reports/actions")
        restored, seen = [], set()

        # lat/lon lookup by geo_region (covers all presets + docker seeds)
        _coords = {
            "asia-south1":          (19.076,  72.877),
            "us-east1":             (38.0,   -78.0),
            "europe-west2":         (51.507,  -0.127),
            "asia-south2":          (28.704,  77.102),
            "asia-southeast1":      (1.352,  103.820),
            "asia-northeast1":      (35.689, 139.692),
            "europe-west3":         (50.110,   8.682),
            "us-west1":             (45.523,-122.675),
            "australia-southeast1": (-33.868, 151.209),
            "southamerica-east1":   (-23.550, -46.633),
        }

        for fpath in _glob.glob(str(reports_dir / "*_actions.jsonl")):
            try:
                with open(fpath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = _json.loads(line)
                        except Exception:
                            continue
                        sid = rec.get("server_id")
                        if not sid or sid in seen:
                            break
                        seen.add(sid)
                        geo_region = rec.get("geo_region", "unknown")
                        geo_label  = rec.get("geo_label", sid)
                        lat, lon   = _coords.get(geo_region, (0.0, 0.0))
                        reg = ServerRegistration(
                            server_id=sid, geo_region=geo_region,
                            geo_label=geo_label, lat=lat, lon=lon,
                            ip="cached", interface="cached", event="register",
                        )
                        registry.register(reg)
                        # Pin heartbeat 1 year out so it never expires
                        registry._last_heartbeat[sid] = (
                            datetime.now(timezone.utc) + timedelta(days=365)
                        )
                        restored.append(sid)
                        break   # one record per file is enough
            except Exception as ex:
                logger.warning(f"[UI] restore-session: error reading {fpath}: {ex}")

        logger.info(f"[UI] Session restore: re-registered {len(restored)} server(s): {restored}")
        
        # ── Group by flow_id to find truly pending threats ──
        merged_records = {}
        for fpath in _glob.glob(str(reports_dir / "*_actions.jsonl")):
            try:
                with open(fpath, "r") as f:
                    for line in f:
                        try:
                            rec = _json.loads(line.strip())
                            fid = rec.get("flow_id")
                            if fid:
                                if fid not in merged_records:
                                    merged_records[fid] = {}
                                merged_records[fid].update(rec)
                        except Exception:
                            pass
            except Exception:
                pass
                
        pending_flows = [
            r for r in merged_records.values() 
            if r.get("council_severity") == "Pending LLM enrichment"
        ]
        
        # ── Inject pending flows directly into the council threat_queue ──
        queued_count = 0
        from src.kafka.schemas import ThreatAlertMessage
        if pipeline and pipeline.threat_queue:
            for rec in pending_flows:
                msg = ThreatAlertMessage(
                    server_id=rec.get("server_id", "unknown"),
                    geo_region=rec.get("geo_region", "unknown"),
                    flow_id=rec.get("flow_id", "unknown"),
                    attack_type=rec.get("attack_type", "Unknown"),
                    confidence=rec.get("confidence", 1.0),
                    severity="Medium",
                    raw_features={} # No features available in cache
                )
                try:
                    pipeline.threat_queue.put_nowait({
                        "msg": msg,
                        "attack_type": msg.attack_type,
                        "confidence": msg.confidence,
                        "X": None
                    })
                    queued_count += 1
                except Exception:
                    pass

        logger.info(f"[UI] Queued {queued_count} pending threats to Council.")
        return {
            "status": "restored",
            "servers_restored": restored,
            "pending_queued": queued_count,
            "total_servers": len(registry.to_api_list()),
        }

    @app.get("/api/pipeline/stream")
    async def pipeline_stream():
        """
        SSE stream of live council/defender events.
        Client sends ?since=<seq> to resume from a known position.
        """
        from fastapi import Request

        async def generator():
            last_seq = 0
            while True:
                events = get_pipeline_events_since(last_seq)
                for ev in events:
                    last_seq = ev["seq"]
                    yield f"data: {json.dumps(ev)}\n\n"
                if not events:
                    await asyncio.sleep(0.3)

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/api/cache")
    def clear_cache(background_tasks: BackgroundTasks):
        """Clear all report files and completely restart the backend process."""
        import glob
        from pathlib import Path
        reports_dir = Path("reports/actions")
        removed = 0
        for fpath in glob.glob(str(reports_dir / "*_actions.jsonl")):
            try:
                Path(fpath).unlink()
                removed += 1
            except Exception:
                pass

        def restart_server():
            import time, sys, os
            time.sleep(1.0)
            logger.warning("[UI] Restarting backend server for fresh session...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        background_tasks.add_task(restart_server)
        return {"status": "cleared", "files_removed": removed, "message": "Server restarting..."}

    # ── Reports API ──────────────────────────────────────────

    @app.get("/api/reports")
    def get_reports(limit: int = 50, server: str = None):
        """
        Read latest action reports from reports/actions/*.jsonl files.
        Returns the most recent `limit` records (newest first) plus summary stats.
        """
        import glob, json as _json
        from pathlib import Path

        reports_dir = Path("reports/actions")
        all_records = []
        lines_to_parse = []

        pattern = str(reports_dir / f"{server}_actions.jsonl") if server else str(reports_dir / "*_actions.jsonl")
        for fpath in glob.glob(pattern):
            try:
                with open(fpath, "r") as f:
                    lines = f.readlines()
                # Use a larger limit when reading lines because some lines are patches, not full records.
                # E.g. limit=50 means we want 50 full records, so read 150 lines to be safe.
                for line in lines[-(limit*3):]:
                    line = line.strip()
                    if line:
                        lines_to_parse.append(line)
            except Exception:
                pass

        # Group by flow_id to merge council_update patches
        merged_records = {}
        for line in lines_to_parse:
            try:
                rec = _json.loads(line)
                fid = rec.get("flow_id")
                if not fid: continue
                
                if rec.get("_type") == "council_update":
                    if fid in merged_records:
                        # Map council_ fields to UI fields
                        if "council_signature" in rec:
                            merged_records[fid]["signature_match"] = rec["council_signature"]
                        if "council_threat_actor" in rec:
                            merged_records[fid]["threat_actor_type"] = rec["council_threat_actor"]
                        if "council_fp_risk" in rec:
                            merged_records[fid]["false_positive_risk"] = rec["council_fp_risk"]
                        merged_records[fid].update(rec)
                    else:
                        merged_records[fid] = rec
                else:
                    if fid in merged_records:
                        merged_records[fid].update(rec)
                    else:
                        merged_records[fid] = rec
            except Exception:
                pass
                
        all_records = list(merged_records.values())

        # Sort newest first by timestamp
        all_records.sort(key=lambda r: r.get("timestamp_utc", ""), reverse=True)
        records = all_records[:limit]

        # Summary stats across all loaded records
        actions = [r.get("rl_action", "MONITOR") for r in all_records]
        from collections import Counter
        action_counts = dict(Counter(actions))
        attack_types = [r.get("attack_type", "Unknown") for r in all_records]
        type_counts = dict(Counter(attack_types))

        # Total counts per file
        file_counts = {}
        for fpath in glob.glob(str(reports_dir / "*_actions.jsonl")):
            try:
                sid = Path(fpath).stem.replace("_actions", "")
                with open(fpath) as f:
                    file_counts[sid] = sum(1 for _ in f)
            except Exception:
                pass

        return {
            "records": records,
            "total_in_files": file_counts,
            "action_summary": action_counts,
            "type_summary": type_counts,
            "returned": len(records),
        }

    @app.get("/api/reports/stream")
    async def reports_stream():
        """SSE stream: push new report records as they are written."""
        import glob, json as _json
        from pathlib import Path

        async def generator():
            reports_dir = Path("reports/actions")
            # Track file positions so we only read new lines
            positions = {}
            for fpath in glob.glob(str(reports_dir / "*_actions.jsonl")):
                try:
                    positions[fpath] = Path(fpath).stat().st_size
                except Exception:
                    positions[fpath] = 0

            while True:
                for fpath in glob.glob(str(reports_dir / "*_actions.jsonl")):
                    if fpath not in positions:
                        positions[fpath] = 0
                    try:
                        current_size = Path(fpath).stat().st_size
                        if current_size > positions[fpath]:
                            with open(fpath, "r") as f:
                                f.seek(positions[fpath])
                                new_lines = f.read()
                                positions[fpath] = f.tell()
                            for line in new_lines.strip().splitlines():
                                if line.strip():
                                    try:
                                        rec = _json.loads(line)
                                        yield f"data: {_json.dumps(rec)}\n\n"
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                await asyncio.sleep(1)

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    @app.post("/api/summarize-threat")
    async def summarize_threat(request: Request):
        """
        Generates an Incident Response Plan locally using extractive summarization.
        Uses sumy TextRank to extract key sentences from each agent transcript,
        then templates the structured threat data into a rich markdown report.
        No LLM API call — instant, deterministic, offline-capable.
        """
        try:
            data = await request.json()

            # ── Core threat fields ────────────────────────────────────────────
            flow_id          = data.get("flow_id", "Unknown")
            attack_type      = data.get("attack_type", "Unknown")
            initial_action   = data.get("rl_action", "Unknown")
            final_action     = data.get("final_rl_action") or initial_action
            confidence       = data.get("confidence", 0)
            severity         = data.get("council_severity", "Unknown")
            council_consensus= data.get("council_consensus", 0)
            fp_risk          = data.get("false_positive_risk", "Unknown")
            signature        = data.get("signature_match", "N/A")
            threat_actor     = data.get("threat_actor_type", "Unknown")
            all_indicators   = data.get("all_indicators", [])
            agent_votes      = data.get("agent_votes", {})
            raw_responses    = data.get("raw_responses", {})
            server_id        = data.get("server_id", "Unknown")
            geo_label        = data.get("geo_label", "")
            latency_ms       = data.get("latency_ms", 0)

            # ── Extractive summarizer (sumy TextRank, fallback to first sentences) ──
            def extractive_summary(text: str, n: int = 3) -> str:
                if not text or len(text.strip()) < 60:
                    return "_No transcript available for this agent._"
                try:
                    from sumy.parsers.plaintext import PlaintextParser
                    from sumy.nlp.tokenizers import Tokenizer
                    from sumy.summarizers.text_rank import TextRankSummarizer
                    from sumy.nlp.stemmers import Stemmer
                    from sumy.utils import get_stop_words
                    parser = PlaintextParser.from_string(
                        text.replace('\n', ' '), Tokenizer("english")
                    )
                    stemmer = Stemmer("english")
                    summarizer = TextRankSummarizer(stemmer)
                    summarizer.stop_words = get_stop_words("english")
                    sentences = summarizer(parser.document, n)
                    result = " ".join(str(s) for s in sentences).strip()
                    return result if result else _fallback_summary(text, n)
                except Exception:
                    return _fallback_summary(text, n)

            def _fallback_summary(text: str, n: int = 3) -> str:
                """First-N-sentences fallback when sumy is unavailable."""
                import re
                sents = re.split(r'(?<=[.!?])\s+', text.strip())
                chosen = [s.strip() for s in sents if len(s.strip()) > 20][:n]
                return " ".join(chosen) if chosen else text[:350].strip() + "…"

            # ── Summarize each agent ─────────────────────────────────────────
            analyst_sum  = extractive_summary(raw_responses.get("analyst",  ""), 3)
            engineer_sum = extractive_summary(raw_responses.get("engineer", ""), 3)
            intel_sum    = extractive_summary(raw_responses.get("intel",    ""), 3)

            # ── Derived fields ───────────────────────────────────────────────
            re_evaluated  = final_action != initial_action
            src_ip = flow_id.split('-')[0] if '-' in flow_id else '?'
            dst_ip = flow_id.split('-')[1] if '-' in flow_id and len(flow_id.split('-')) > 1 else '?'
            dst_port = flow_id.split('-')[3] if '-' in flow_id and len(flow_id.split('-')) > 3 else '?'

            indicators_md = "\n".join(f"- `{i}`" for i in all_indicators) \
                            if all_indicators else "- _No specific indicators extracted_"

            action_rationale = (
                f"After the AI Council elevated severity to **{severity}**, "
                f"the RL Defender **upgraded its response** from `{initial_action}` → `{final_action}`."
                if re_evaluated else
                f"The RL Defender **confirmed** its pre-Council decision of `{final_action}`, "
                f"consistent with the Council's **{severity}** classification."
            )

            risk_badge = {
                "Critical": "🔴 **CRITICAL** — Immediate escalation required.",
                "High":     "🟠 **HIGH** — Investigate within 1 hour.",
                "Medium":   "🟡 **MEDIUM** — Investigate during business hours.",
                "Low":      "🟢 **LOW** — Log and monitor. No immediate action required.",
            }.get(severity, "⚪ **UNKNOWN** — Manual review required.")

            votes_md = "\n".join([
                f"| Security Analyst | `{agent_votes.get('analyst', 'N/A')}` |",
                f"| ML Engineer      | `{agent_votes.get('engineer','N/A')}` |",
                f"| Threat Intel     | `{agent_votes.get('intel',   'N/A')}` |",
            ])

            # ── Build the markdown report ─────────────────────────────────────
            report = f"""# Incident Response Plan

> **Agentic-IDS** · Server `{server_id}` {("· " + geo_label) if geo_label else ""} · E2E latency `{latency_ms:.0f}ms`

---

## 1. Threat Summary

The ML Ensemble (XGBoost + LSTM) classified flow `{flow_id}` as a **{attack_type}** attack with **{confidence*100:.1f}%** confidence. The three-agent AI Council reached **{council_consensus*100:.0f}% consensus** and classified severity as **{severity}**. The RL Defender responded with **`{final_action}`**.

---

## 2. Attack Path

| Field | Value |
|---|---|
| Source IP | `{src_ip}` |
| Destination | `{dst_ip}:{dst_port}` |
| Attack Type | **{attack_type}** |
| ML Confidence | **{confidence*100:.1f}%** |
| Signature Match | `{signature}` |
| Threat Actor Type | {threat_actor} |
| FP Risk | **{fp_risk}** |

---

## 3. AI Council Deliberation

| Agent | Vote |
|---|---|
{votes_md}

**Security Analyst** (LM Studio local model):
> {analyst_sum}

**ML Engineer** (Ollama local model):
> {engineer_sum}

**Threat Intel** (Groq cloud model):
> {intel_sum}

---

## 4. Defender Actions

{action_rationale}

| Stage | Action |
|---|---|
| RL Stage 1 — Pre-Council  | `{initial_action}` |
| RL Stage 2 — Post-Council | `{final_action}` |

---

## 5. Threat Indicators

{indicators_md}

---

## 6. Recommended Next Steps

1. **Verify** the `{final_action}` action was applied successfully on `{dst_ip}:{dst_port}`
2. **Investigate** source `{src_ip}` for prior threat history and lateral movement
3. **Review** detection signatures for **{attack_type}** patterns to reduce false positives (FP risk: `{fp_risk}`)
4. **Update** blocklists and IDS rules based on the indicators above
5. **Escalate** to SOC Tier 2 if the threat persists or more flows from `{src_ip}` are detected

---

## 7. Risk Assessment

{risk_badge}

> Council Consensus: **{council_consensus*100:.0f}%** · Severity: **{severity}** · FP Risk: **{fp_risk}**
"""
            return {"status": "success", "summary": report}

        except Exception as e:
            logger.error(f"Error summarizing threat: {e}")
            return {"status": "error", "message": str(e)}


    # ── Dashboard HTML ────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        html_path = os.path.join(_FRONTEND_DIR, "index.html")
        if os.path.exists(html_path):
            with open(html_path) as f:
                return f.read()
        return HTMLResponse(
            f"<h1>Agentic-IDS Server Manager</h1>"
            f"<p>Frontend not found at: {html_path}</p>"
            f"<p>Make sure frontend/index.html exists at the repo root.</p>"
        )


    # ─────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────

    def _start_replay(req: AddServerRequest, reg: ServerRegistration):
        """Slice the dataset for this server and start Kafka replay."""
        try:
            from src.simulation.dataset_slicer import DatasetSlicer, ServerSlice
            from src.simulation.replay_engine import ReplayEngine

            slicer = DatasetSlicer(req.data_path, n_servers=req.n_servers_total)
            slices = slicer.slice()

            # Find the slice for this server (by partition index)
            slice_idx = reg.kafka_partition % len(slices)
            server_slice = slices[slice_idx]
            # Override server metadata to match registration
            server_slice.server_id = reg.server_id
            server_slice.geo_region = reg.geo_region
            server_slice.geo_label = reg.geo_label
            server_slice.lat = reg.lat
            server_slice.lon = reg.lon

            engine = ReplayEngine(server_slice, flows_per_second=req.flows_per_second)
            bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            engine.start_kafka_replay(bootstrap_servers=bootstrap)
            active_replays[reg.server_id] = engine

            logger.info(
                f"[UI] Started replay for {reg.server_id}: "
                f"{len(server_slice.df)} flows @ {req.flows_per_second}/s"
            )
        except Exception as e:
            logger.error(f"[UI] Failed to start replay for {reg.server_id}: {e}")
