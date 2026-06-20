"""
Kernel Action Executor
======================
Translates RL defender decisions into concrete, geo-aware mitigation commands.

Architecture:
  DefenderRLAgent.act() → KernelExecutor.execute() → ActionReport (JSON)

Modes:
  DRY_RUN=True  (default) — generates the exact commands that *would* run,
                             logs them, appends to report file. No OS calls.
  DRY_RUN=False            — fires subprocess commands (requires sudo/CAP_NET_ADMIN).

Action → Command mapping:
  BLOCK_SOURCE       → iptables -I INPUT -s <src_ip> -j DROP
  RATE_LIMIT         → iptables hashlimit rule limiting pps from source
  DEEP_PACKET_INSPECTION → iptables NFQUEUE redirect + nfqueue listener spawn
  MONITOR            → tcpdump filter expression (no kernel modification)

Each report entry contains:
  - flow_id, server_id, geo_region, geo_label
  - attack_type, confidence (from ML ensemble)
  - action decided by RL agent
  - council_recommendation (from LLM enrichment, may be None if still running)
  - exact commands that would/did execute
  - timestamp, dry_run flag, execution status
"""

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Geo metadata for human-readable reports ─────────────────────────────────

GEO_LABELS = {
    "asia-south1":   "Mumbai, India",
    "us-east1":      "Virginia, USA",
    "europe-west2":  "London, UK",
    "ap-southeast1": "Singapore",
    "us-west1":      "Oregon, USA",
}

# ─── iptables command templates ───────────────────────────────────────────────

_IPTABLES = "iptables"   # swap to "ip6tables" for IPv6

# Block all traffic from this source IP
_CMD_BLOCK = [
    _IPTABLES, "-I", "INPUT", "-s", "{src_ip}", "-j", "DROP",
]
_CMD_BLOCK_COMMENT = (
    "iptables -I INPUT -s {src_ip} -j DROP  "
    "# hard block — all packets from {src_ip} dropped at kernel level"
)

# Rate-limit: allow up to 50 packets/sec from source, drop the rest
_CMD_RATE_LIMIT = [
    _IPTABLES, "-A", "INPUT",
    "-s", "{src_ip}",
    "-m", "hashlimit",
    "--hashlimit-name", "ids_rl_{flow_id}",
    "--hashlimit-above", "50/sec",
    "--hashlimit-mode", "srcip",
    "--hashlimit-burst", "100",
    "-j", "DROP",
]
_CMD_RATE_LIMIT_COMMENT = (
    "iptables hashlimit: allow ≤50 pkt/s from {src_ip}, drop excess  "
    "# soft throttle preserving legitimate traffic"
)

# Redirect traffic to NFQUEUE for userspace DPI (Suricata / custom nfqueue listener)
_CMD_DPI_NFQUEUE = [
    _IPTABLES, "-A", "INPUT",
    "-s", "{src_ip}",
    "-j", "NFQUEUE", "--queue-num", "0",
]
_CMD_DPI_NFQUEUE_COMMENT = (
    "iptables -j NFQUEUE: redirect {src_ip} → queue 0 for deep packet inspection  "
    "# requires nfqueue listener (e.g. Suricata in IPS mode)"
)

# MONITOR: no kernel change — generate tcpdump capture filter for the analyst
_CMD_MONITOR_COMMENT = (
    "tcpdump -i any 'host {src_ip}' -w /tmp/capture_{flow_id}.pcap  "
    "# passive capture only — no traffic modification"
)


class ActionReport:
    """A single defender decision record, serialisable to JSON."""

    __slots__ = (
        "flow_id", "server_id", "geo_region", "geo_label",
        "attack_type", "confidence",
        "rl_action", "rl_action_id",
        "council_recommendation", "council_severity", "council_consensus",
        "src_ip",
        "commands", "command_comments",
        "dry_run", "executed", "execution_error",
        "timestamp_utc", "latency_ms",
    )

    def to_dict(self) -> dict:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __str__(self) -> str:
        status = "DRY-RUN" if self.dry_run else ("OK" if self.executed else "FAILED")
        return (
            f"[Executor] {self.server_id} ({self.geo_label}) | "
            f"{self.attack_type} ({self.confidence:.1%}) → {self.rl_action} | "
            f"{status}"
        )


class KernelExecutor:
    """
    Geo-aware RL action executor.

    Parameters
    ----------
    report_dir : str
        Directory to write per-server JSON report files.
        Each server gets its own file: <report_dir>/<server_id>_actions.jsonl
    dry_run : bool
        If True (default), commands are built but never executed.
        Set to False only in a privileged Linux environment with CAP_NET_ADMIN.
    """

    def __init__(
        self,
        report_dir: str = "reports/actions",
        dry_run: bool = True,
    ):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run

        mode = "DRY-RUN (commands logged, not executed)" if dry_run else "LIVE (kernel calls enabled)"
        logger.info(f"[Executor] Initialized — mode: {mode}")
        logger.info(f"[Executor] Reports directory: {self.report_dir.resolve()}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def execute(
        self,
        *,
        flow_id: str,
        server_id: str,
        geo_region: str,
        attack_type: str,
        confidence: float,
        rl_action: str,
        rl_action_id: int,
        latency_ms: float,
        council_result: Optional[dict] = None,
        src_ip: Optional[str] = None,
    ) -> ActionReport:
        """
        Build the action report, optionally execute kernel commands, persist to disk.

        Parameters
        ----------
        council_result : dict or None
            ThreatAnalysis.to_dict() from the LLM council, or None if enrichment
            is still in progress (fire-and-forget background task).
        src_ip : str or None
            Source IP extracted from flow features. Many CSV-sourced datasets don't
            carry a raw IP — pass None and the executor will note it as unavailable.
        """
        report = ActionReport()
        report.flow_id      = flow_id
        report.server_id    = server_id
        report.geo_region   = geo_region
        report.geo_label    = GEO_LABELS.get(geo_region, geo_region)
        report.attack_type  = attack_type
        report.confidence   = confidence
        report.rl_action    = rl_action
        report.rl_action_id = rl_action_id
        report.latency_ms   = latency_ms
        report.src_ip       = src_ip or "N/A (not in flow features)"
        report.dry_run      = self.dry_run
        report.timestamp_utc = datetime.now(timezone.utc).isoformat()

        # ── Council enrichment (may be None if LLM still running) ────────────
        if council_result and isinstance(council_result, dict):
            report.council_recommendation = council_result.get("recommendations", [])
            report.council_severity       = council_result.get("severity", "Unknown")
            report.council_consensus      = council_result.get("council_consensus", 0.0)
        elif hasattr(council_result, "to_dict"):
            # ThreatAnalysis object (not yet a dict)
            d = council_result.to_dict()
            report.council_recommendation = d.get("recommendations", [])
            report.council_severity       = d.get("severity", "Unknown")
            report.council_consensus      = d.get("council_consensus", 0.0)
        else:
            report.council_recommendation = []
            report.council_severity       = "Pending LLM enrichment"
            report.council_consensus      = 0.0

        # ── Build commands ────────────────────────────────────────────────────
        commands, comments = self._build_commands(rl_action, src_ip, flow_id)
        report.commands         = commands
        report.command_comments = comments

        # ── Execute (or dry-run) ──────────────────────────────────────────────
        report.executed        = False
        report.execution_error = None

        if not self.dry_run and src_ip and src_ip != "N/A (not in flow features)":
            report.executed, report.execution_error = self._run_commands(commands)
        else:
            if not self.dry_run and not src_ip:
                report.execution_error = (
                    "Skipped: src_ip not available in flow features — "
                    "cannot target iptables rule without IP address."
                )

        # ── Log ──────────────────────────────────────────────────────────────
        logger.info(str(report))
        for comment in comments:
            logger.info(f"  CMD: {comment}")

        # ── Persist ──────────────────────────────────────────────────────────
        self._persist(report)

        return report

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_commands(
        self,
        action: str,
        src_ip: Optional[str],
        flow_id: str,
    ) -> tuple[list[list[str]], list[str]]:
        """Return (argv_lists, human_readable_comments) for the chosen action."""
        ip = src_ip or "<src_ip_unavailable>"
        fid = flow_id.replace(":", "_").replace("/", "_")[:16]

        if action == "BLOCK_SOURCE":
            argv = [[a.format(src_ip=ip) for a in _CMD_BLOCK]]
            comments = [_CMD_BLOCK_COMMENT.format(src_ip=ip)]

        elif action == "RATE_LIMIT":
            argv = [[a.format(src_ip=ip, flow_id=fid) for a in _CMD_RATE_LIMIT]]
            comments = [_CMD_RATE_LIMIT_COMMENT.format(src_ip=ip)]

        elif action == "DEEP_PACKET_INSPECTION":
            argv = [[a.format(src_ip=ip) for a in _CMD_DPI_NFQUEUE]]
            comments = [_CMD_DPI_NFQUEUE_COMMENT.format(src_ip=ip, flow_id=fid)]

        else:  # MONITOR
            argv = []  # no kernel change
            comments = [_CMD_MONITOR_COMMENT.format(src_ip=ip, flow_id=fid)]

        return argv, comments

    def _run_commands(self, argv_lists: list[list[str]]) -> tuple[bool, Optional[str]]:
        """Execute each command via subprocess. Returns (success, error_message)."""
        for argv in argv_lists:
            try:
                result = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode != 0:
                    return False, f"Command failed ({result.returncode}): {result.stderr.strip()}"
            except subprocess.TimeoutExpired:
                return False, "Command timed out after 5s"
            except FileNotFoundError as e:
                return False, f"Binary not found: {e}"
            except Exception as e:
                return False, str(e)
        return True, None

    def _persist(self, report: ActionReport) -> None:
        """Append report as a JSON line to the per-server report file."""
        path = self.report_dir / f"{report.server_id}_actions.jsonl"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(report.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"[Executor] Failed to write report: {e}")
