"""
Human feedback loop for analyst-in-the-loop mitigation decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class HumanFeedbackDecision:
    """Structured record of an analyst review."""

    reviewed: bool
    approved: bool
    action_id: int
    action: str
    verdict: str
    notes: str
    reviewer: str
    timestamp: str
    reason: str = ""


class HumanFeedbackLoop:
    """Collect and persist human feedback before automated mitigation."""

    def __init__(
        self,
        mode: str = "auto",
        confidence_threshold: float = 0.6,
        log_path: str = "logs/human_feedback.jsonl",
        reviewer: str = "human_analyst",
    ):
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.log_path = log_path
        self.reviewer = reviewer
        self.actions = {
            0: "MONITOR",
            1: "BLOCK_SOURCE",
            2: "DEEP_PACKET_INSPECTION",
            3: "RATE_LIMIT",
        }

    def should_review(self, confidence: float) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "always":
            return True
        return confidence >= self.confidence_threshold

    def review(
        self,
        flow_id: str,
        flow_data: Dict[str, Any],
        prediction: Dict[str, Any],
        council_result: Any,
        suggested_action: Dict[str, Any],
    ) -> HumanFeedbackDecision:
        """Return a human decision, or an auto-approval when review is unavailable."""

        confidence = float(prediction.get("confidence", 0.0))
        action_id = int(suggested_action.get("action_id", 0))
        action_name = suggested_action.get("action", self.actions.get(action_id, "MONITOR"))

        if not self.should_review(confidence):
            decision = self._decision(False, True, action_id, action_name, "not_reviewed", "", "below_threshold")
            self._persist(flow_id, flow_data, prediction, council_result, suggested_action, decision)
            return decision

        if not sys.stdin.isatty():
            decision = self._decision(False, True, action_id, action_name, "not_reviewed", "", "non_interactive")
            logger.info("Human review requested, but stdin is non-interactive. Auto-approving suggested action.")
            self._persist(flow_id, flow_data, prediction, council_result, suggested_action, decision)
            return decision

        try:
            decision = self._prompt_for_decision(prediction, council_result, suggested_action)
        except EOFError:
            decision = self._decision(False, True, action_id, action_name, "not_reviewed", "", "input_unavailable")
            logger.info("Human review input unavailable. Auto-approving suggested action.")
        self._persist(flow_id, flow_data, prediction, council_result, suggested_action, decision)
        return decision

    def _prompt_for_decision(
        self,
        prediction: Dict[str, Any],
        council_result: Any,
        suggested_action: Dict[str, Any],
    ) -> HumanFeedbackDecision:
        logger.info("\n" + "=" * 60)
        logger.info("HUMAN FEEDBACK REQUIRED")
        logger.info("=" * 60)
        logger.info(f"Detector: {prediction.get('attack_type', 'Unknown')} ({float(prediction.get('confidence', 0.0)):.1%} conf)")
        logger.info(f"Council severity: {self._safe_get(council_result, 'severity', 'Unknown')}")
        logger.info(f"Suggested action: {suggested_action.get('action', 'MONITOR')}")
        logger.info("Choose: [a]pprove, [m]onitor, [b]lock, [d]pi, [r]ate-limit, [x] reject")

        choice_map = {
            "a": (True, int(suggested_action.get("action_id", 0)), suggested_action.get("action", "MONITOR")),
            "m": (True, 0, "MONITOR"),
            "b": (True, 1, "BLOCK_SOURCE"),
            "d": (True, 2, "DEEP_PACKET_INSPECTION"),
            "r": (True, 3, "RATE_LIMIT"),
            "x": (False, 0, "MONITOR"),
        }

        choice = ""
        while choice not in choice_map:
            choice = input("Analyst decision: ").strip().lower()[:1]

        verdict = input("Verdict [true_positive/false_positive/uncertain]: ").strip() or "uncertain"
        notes = input("Notes: ").strip()
        approved, action_id, action_name = choice_map[choice]
        reason = "human_approved" if approved else "human_rejected"
        return self._decision(True, approved, action_id, action_name, verdict, notes, reason)

    def _decision(
        self,
        reviewed: bool,
        approved: bool,
        action_id: int,
        action: str,
        verdict: str,
        notes: str,
        reason: str,
    ) -> HumanFeedbackDecision:
        return HumanFeedbackDecision(
            reviewed=reviewed,
            approved=approved,
            action_id=action_id,
            action=action,
            verdict=verdict,
            notes=notes,
            reviewer=self.reviewer,
            timestamp=datetime.now().isoformat(),
            reason=reason,
        )

    def _persist(
        self,
        flow_id: str,
        flow_data: Dict[str, Any],
        prediction: Dict[str, Any],
        council_result: Any,
        suggested_action: Dict[str, Any],
        decision: HumanFeedbackDecision,
    ) -> None:
        record = {
            "flow_id": flow_id,
            "prediction": self._json_safe(prediction),
            "council": self._summarize_council(council_result),
            "suggested_action": self._json_safe(suggested_action),
            "human_feedback": asdict(decision),
            "flow_snapshot": self._json_safe(dict(list(flow_data.items())[:20])),
        }

        try:
            log_dir = os.path.dirname(self.log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to persist human feedback: {exc}")

    def _summarize_council(self, council_result: Any) -> Dict[str, Any]:
        return {
            "threat_type": self._safe_get(council_result, "threat_type", "Unknown"),
            "severity": self._safe_get(council_result, "severity", "Unknown"),
            "consensus": self._safe_get(
                council_result,
                "council_consensus",
                self._safe_get(council_result, "consensus_score", "N/A"),
            ),
            "recommendations": self._safe_get(council_result, "recommendations", []),
            "timestamp": self._safe_get(council_result, "timestamp", ""),
        }

    def _safe_get(self, obj: Any, key: str, default: Optional[Any] = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _json_safe(self, value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except TypeError:
            if hasattr(value, "item"):
                return value.item()
            if isinstance(value, dict):
                return {str(k): self._json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [self._json_safe(v) for v in value]
            return str(value)
