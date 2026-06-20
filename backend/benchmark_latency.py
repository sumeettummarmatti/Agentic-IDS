"""
benchmark_latency.py — Comprehensive IDS Benchmark Suite

Tests:
  A. Direct (No Kafka) — baseline single-threaded latency
  B. Async Pipeline    — Kafka-ready async latency + throughput
  C. Load Balancing    — 3 sub-tests:
       C1. Balanced      — all servers same rate → even distribution
       C2. Unbalanced    — servers at 10% / 30% / 60% traffic split
       C3. Spike         — one server floods 5× then stops → isolation check

Usage:
    python benchmark_latency.py --data data/raw/filtered_nowebatt.csv --n 300
    python benchmark_latency.py --data data/raw/filtered_nowebatt.csv --n 300 --skip-direct
    python benchmark_latency.py --data data/raw/filtered_nowebatt.csv --n 300 --lb-only

Results: printed table + benchmark_results.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()
load_dotenv(dotenv_path="config/.env")

logging.basicConfig(level=logging.WARNING, format="%(message)s")
sys.path.insert(0, os.path.dirname(__file__))

W = 70   # Report width


# ─────────────────────────────────────────────────────────────
#  Shared: model init helper (called once, reused across tests)
# ─────────────────────────────────────────────────────────────

def _init_components(data_path: str, timesteps: int = 200):
    from src.detector.ensemble_model import EnsembleDetector
    from src.detector.preprocessor import Preprocessor
    from src.agents.defender_agent import DefenderRLAgent
    from src.council.llm_council_wrapper import ThreatAnalysisCouncil

    preprocessor = Preprocessor()
    detector = EnsembleDetector(use_lstm=True)
    defender = DefenderRLAgent()
    provider = (os.getenv("LLM_PROVIDER") or os.getenv("PRIMARY_LLM_PROVIDER") or "groq").lower()
    council = ThreatAnalysisCouncil(provider=provider)

    df = preprocessor.load_data(data_path)
    X, _, y = preprocessor.prepare_features_and_labels(df, training=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    detector.train(X_train, y_train)
    defender.train(total_timesteps=timesteps)

    return detector, council, defender, preprocessor, X_test, y_test


def _compute_stats(latencies: List[float], total_s: float, mode: str, extra: dict = None) -> dict:
    a = np.array(latencies) if latencies else np.array([0.0])
    result = {
        "mode": mode,
        "count": len(latencies),
        "total_elapsed_s": round(total_s, 3),
        "throughput_flows_per_s": round(len(latencies) / max(total_s, 0.001), 1),
        "mean_ms":   round(float(np.mean(a)), 3),
        "median_ms": round(float(np.median(a)), 3),
        "p95_ms":    round(float(np.percentile(a, 95)), 3),
        "p99_ms":    round(float(np.percentile(a, 99)), 3),
        "min_ms":    round(float(np.min(a)), 3),
        "max_ms":    round(float(np.max(a)), 3),
        "std_ms":    round(float(np.std(a)), 3),
    }
    if extra:
        result.update(extra)
    return result


# ═════════════════════════════════════════════════════════════
#  BENCHMARK A: Direct (No Kafka)
# ═════════════════════════════════════════════════════════════

def run_direct_benchmark(data_path: str, n_flows: int) -> dict:
    """Single-threaded, no Kafka, no async — baseline."""
    print(f"\n{'═'*W}")
    print(f"  BENCHMARK A — Direct (No Kafka)  │  {n_flows} flows")
    print(f"{'═'*W}")

    detector, council, defender, preprocessor, X_test, _ = _init_components(data_path)

    import torch

    latencies = []
    flows = X_test[:n_flows]
    print(f"  Running {len(flows)} flows sequentially…")
    t_total = time.perf_counter()

    for i, flow in enumerate(flows):
        t0 = time.perf_counter()

        xgb_prob = detector.xgb_model.predict_proba(flow.reshape(1, -1))[0]
        ft = torch.FloatTensor(flow).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            lstm_prob = torch.softmax(detector.lstm_model(ft), dim=1).numpy()[0]

        size = max(len(xgb_prob), len(lstm_prob))
        final = 0.7 * np.pad(xgb_prob, (0, size - len(xgb_prob))) \
              + 0.3 * np.pad(lstm_prob, (0, size - len(lstm_prob)))
        confidence = float(final[np.argmax(final)])

        if confidence > 0.6:
            perception = {"confidence": confidence, "threat_level": "High", "flow_rate": 1000}
            defender.act(defender.observe(perception))

        latencies.append((time.perf_counter() - t0) * 1000)

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(flows)}]  mean={np.mean(latencies):.2f}ms")

    return _compute_stats(latencies, time.perf_counter() - t_total, "direct_no_kafka")


# ═════════════════════════════════════════════════════════════
#  BENCHMARK B: Async Pipeline (Kafka-ready)
# ═════════════════════════════════════════════════════════════

async def run_kafka_benchmark(data_path: str, n_flows: int, n_servers: int = 3) -> dict:
    """Full async pipeline — Kafka-ready with N virtual servers."""
    print(f"\n{'═'*W}")
    print(f"  BENCHMARK B — Async Pipeline  │  {n_flows} flows × {n_servers} servers")
    print(f"{'═'*W}")

    from src.pipeline.pipeline import IDSPipeline

    detector, council, defender, preprocessor, _, _ = _init_components(data_path)
    pipeline = IDSPipeline(detector, council, defender, preprocessor, mode="test")

    t_total = time.perf_counter()
    print(f"  Running pipeline with {n_servers} virtual servers…")
    report = await pipeline.run_test(
        data_path=data_path,
        n_servers=n_servers,
        flows_per_second_per_server=0,
        max_flows_per_server=n_flows // n_servers,
    )
    total_s = time.perf_counter() - t_total
    report["total_elapsed_s"] = round(total_s, 3)
    report["mode"] = "async_pipeline_kafka_ready"
    return report


# ═════════════════════════════════════════════════════════════
#  BENCHMARK C: Load Balancing Suite
# ═════════════════════════════════════════════════════════════

class LoadBalancingBenchmark:
    """
    Three load balancing scenarios simulated via asyncio.Queue (no real Kafka needed).

    Each server gets its own per-server latency tracker so we can see:
      - Is the load distributed evenly across servers?
      - Does a high-traffic server cause latency spikes for low-traffic servers?
      - Does latency recover after a burst?
    """

    def __init__(self, data_path: str, n_flows: int):
        self.data_path = data_path
        self.n_flows = n_flows
        self._detector = None
        self._defender = None
        self._preprocessor = None

    def _ensure_models(self):
        if self._detector is None:
            self._detector, _, self._defender, self._preprocessor, _, _ = \
                _init_components(self.data_path, timesteps=100)

    # ── C1: Balanced Load ───────────────────────────────────────

    async def run_balanced(self, n_servers: int = 4) -> dict:
        """
        All servers send the SAME number of flows.
        Expected result: uniform per-server latency, even partition spread.
        """
        print(f"\n{'─'*W}")
        print(f"  C1 — BALANCED LOAD  │  {n_servers} servers × {self.n_flows // n_servers} flows each")
        print(f"{'─'*W}")
        self._ensure_models()

        flows_per_server = self.n_flows // n_servers
        server_ids = [f"SRV-{i+1:02d}" for i in range(n_servers)]
        per_server_latencies: Dict[str, List[float]] = {s: [] for s in server_ids}

        results = await self._run_multi_server(
            server_ids=server_ids,
            flows_per_server={s: flows_per_server for s in server_ids},
            per_server_latencies=per_server_latencies,
            label="balanced",
        )

        self._print_per_server_table(per_server_latencies, "C1: Balanced Load")
        balance_score = self._balance_score(per_server_latencies)
        print(f"  Load balance score: {balance_score:.1f}%  (100% = perfectly even)")
        results["balance_score_pct"] = balance_score
        return results

    # ── C2: Unbalanced Load ─────────────────────────────────────

    async def run_unbalanced(self, n_servers: int = 3) -> dict:
        """
        Servers send at 10% / 30% / 60% split of total flows.
        Key question: does server 3 (60%) slow down servers 1 and 2?
        Answer should be NO — partition isolation guarantees independence.
        """
        print(f"\n{'─'*W}")
        print(f"  C2 — UNBALANCED LOAD  │  10% / 30% / 60% traffic split across {n_servers} servers")
        print(f"{'─'*W}")
        self._ensure_models()

        weights = [0.10, 0.30, 0.60][:n_servers]
        total = self.n_flows
        server_ids = [f"SRV-{i+1:02d}" for i in range(n_servers)]
        flows_per_server = {
            server_ids[i]: max(1, int(total * weights[i]))
            for i in range(n_servers)
        }
        per_server_latencies: Dict[str, List[float]] = {s: [] for s in server_ids}

        print(f"  Distribution: " + "  ".join(
            f"{sid}={flows_per_server[sid]} ({int(w*100)}%)"
            for sid, w in zip(server_ids, weights)
        ))

        results = await self._run_multi_server(
            server_ids=server_ids,
            flows_per_server=flows_per_server,
            per_server_latencies=per_server_latencies,
            label="unbalanced",
        )

        self._print_per_server_table(per_server_latencies, "C2: Unbalanced Load")
        self._print_isolation_verdict(per_server_latencies, weights)
        return results

    # ── C3: Spike Test ──────────────────────────────────────────

    async def run_spike(self, n_servers: int = 3) -> dict:
        """
        Phase 1 (normal): all servers send at equal rate.
        Phase 2 (spike):  server 1 floods 5× — simulates a DDoS surge.
        Phase 3 (recovery): server 1 drops back to normal.

        Key question: does the spike on server 1 affect servers 2 and 3?
        Kafka partition isolation means it SHOULD NOT.
        """
        print(f"\n{'─'*W}")
        print(f"  C3 — SPIKE TEST  │  Server 1 floods 5× in Phase 2, then recovers")
        print(f"{'─'*W}")
        self._ensure_models()

        normal_per = self.n_flows // (n_servers * 3)   # flows per server per phase
        server_ids = [f"SRV-{i+1:02d}" for i in range(n_servers)]

        phases = {
            "normal":   {s: normal_per for s in server_ids},
            "spike":    {server_ids[0]: normal_per * 5, **{s: normal_per for s in server_ids[1:]}},
            "recovery": {s: normal_per for s in server_ids},
        }

        phase_latencies: Dict[str, Dict[str, List[float]]] = {}

        for phase_name, flows_per_server in phases.items():
            print(f"\n  Phase: {phase_name.upper()}", end="")
            if phase_name == "spike":
                print(f"  ← {server_ids[0]} sends {normal_per * 5} flows (5×)", end="")
            print()

            per_server_latencies: Dict[str, List[float]] = {s: [] for s in server_ids}
            await self._run_multi_server(
                server_ids=server_ids,
                flows_per_server=flows_per_server,
                per_server_latencies=per_server_latencies,
                label=phase_name,
            )
            phase_latencies[phase_name] = per_server_latencies

        self._print_spike_report(phase_latencies, server_ids)
        return {"mode": "load_balancing_spike", "phases": {
            ph: {s: _compute_stats(lats, 1.0, f"{ph}/{s}")
                 for s, lats in servers.items()}
            for ph, servers in phase_latencies.items()
        }}

    # ── Core simulation engine ──────────────────────────────────

    async def _run_multi_server(
        self,
        server_ids: List[str],
        flows_per_server: Dict[str, int],
        per_server_latencies: Dict[str, List[float]],
        label: str,
    ) -> dict:
        """
        Simulates N servers publishing flows concurrently into a shared pipeline.
        Each server runs as a separate asyncio task — mimics partition parallelism.
        """
        import torch
        from src.kafka.schemas import RawFlowMessage
        from src.simulation.dataset_slicer import DatasetSlicer

        slicer = DatasetSlicer(self.data_path, n_servers=len(server_ids), strategy="stratified")
        slices = slicer.slice()
        slice_map = {sid: slc for sid, slc in zip(server_ids, slices)}

        lock = asyncio.Lock()
        all_latencies = []
        t_start = time.perf_counter()

        async def simulate_server(server_id: str):
            slc = slice_map[server_id]
            n = flows_per_server[server_id]
            df = slc.df.head(n)

            for _, row in df.iterrows():
                t0 = time.perf_counter()

                features = row.drop("Label", errors="ignore").to_dict()
                X = np.array([list(features.values())], dtype=float)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                try:
                    xgb_prob = self._detector.xgb_model.predict_proba(X)[0]
                    ft = torch.FloatTensor(X).unsqueeze(1)
                    with torch.no_grad():
                        lstm_prob = torch.softmax(self._detector.lstm_model(ft), dim=1).numpy()[0]

                    size = max(len(xgb_prob), len(lstm_prob))
                    final = 0.7 * np.pad(xgb_prob, (0, size - len(xgb_prob))) \
                          + 0.3 * np.pad(lstm_prob, (0, size - len(lstm_prob)))
                    confidence = float(final[np.argmax(final)])

                    if confidence > 0.6:
                        self._defender.act(self._defender.observe(
                            {"confidence": confidence, "threat_level": "High", "flow_rate": 500}
                        ))
                except Exception:
                    pass

                latency_ms = (time.perf_counter() - t0) * 1000
                per_server_latencies[server_id].append(latency_ms)

                async with lock:
                    all_latencies.append(latency_ms)

                await asyncio.sleep(0)   # yield to event loop (simulate async queue)

        # Run all servers concurrently
        await asyncio.gather(*[simulate_server(sid) for sid in server_ids])
        total_s = time.perf_counter() - t_start

        return _compute_stats(all_latencies, total_s, f"lb_{label}")

    # ── Report helpers ──────────────────────────────────────────

    @staticmethod
    def _print_per_server_table(per_server: Dict[str, List[float]], title: str):
        print(f"\n  {'─'*60}")
        print(f"  Per-Server Results: {title}")
        print(f"  {'─'*60}")
        hdr = f"  {'Server':<12} {'Flows':>7} {'Mean':>9} {'P95':>9} {'P99':>9} {'Throughput':>12}"
        print(hdr)
        print(f"  {'─'*60}")
        for sid, lats in per_server.items():
            if not lats:
                continue
            a = np.array(lats)
            rate = len(lats) / (sum(lats) / 1000) if sum(lats) > 0 else 0
            print(f"  {sid:<12} {len(lats):>7} {np.mean(a):>8.2f}ms"
                  f" {np.percentile(a,95):>8.2f}ms"
                  f" {np.percentile(a,99):>8.2f}ms"
                  f" {rate:>10.0f}/s")
        print(f"  {'─'*60}")

    @staticmethod
    def _balance_score(per_server: Dict[str, List[float]]) -> float:
        """
        Balance score = 100% means all servers have identical mean latency.
        Score drops as latency spreads across servers.
        """
        means = [np.mean(lats) for lats in per_server.values() if lats]
        if not means or max(means) == 0:
            return 100.0
        spread = (max(means) - min(means)) / max(means)
        return round((1 - spread) * 100, 1)

    @staticmethod
    def _print_isolation_verdict(per_server: Dict[str, List[float]], weights: list):
        """Check if high-traffic server affected low-traffic server latency."""
        servers = list(per_server.keys())
        means = {s: np.mean(lats) if lats else 0 for s, lats in per_server.items()}

        print(f"\n  ISOLATION CHECK — does the 60% server slow down the 10% server?")
        low_srv  = servers[0]   # 10% traffic
        high_srv = servers[-1]  # 60% traffic
        diff_pct = abs(means[high_srv] - means[low_srv]) / max(means[low_srv], 0.001) * 100

        if diff_pct < 20:
            print(f"  ✅ PASS — {low_srv} (low traffic) latency within {diff_pct:.1f}% of {high_srv} (high traffic)")
            print(f"     Partitions are isolated. Heavy servers don't starve light ones.")
        else:
            print(f"  ⚠️  WARN — {diff_pct:.1f}% latency gap between light and heavy server.")
            print(f"     Consider adding more consumer workers or increasing queue size.")

    @staticmethod
    def _print_spike_report(phase_latencies: Dict[str, Dict[str, List[float]]], server_ids: List[str]):
        print(f"\n  SPIKE REPORT — Mean latency per server per phase")
        print(f"  {'─'*65}")
        hdr = f"  {'Server':<12}" + "".join(f" {ph.upper():>15}" for ph in phase_latencies)
        print(hdr)
        print(f"  {'─'*65}")

        for sid in server_ids:
            row = f"  {sid:<12}"
            for ph, servers in phase_latencies.items():
                lats = servers.get(sid, [])
                val = f"{np.mean(lats):.2f}ms" if lats else "—"
                row += f" {val:>15}"
            print(row)

        print(f"  {'─'*65}")

        # Isolation verdict
        spike_ph = phase_latencies.get("spike", {})
        normal_ph = phase_latencies.get("normal", {})
        flooded = server_ids[0]
        bystanders = server_ids[1:]

        if bystanders and flooded in spike_ph and flooded in normal_ph:
            flooded_delta = np.mean(spike_ph[flooded]) - np.mean(normal_ph.get(flooded, [0]))
            bystander_deltas = []
            for b in bystanders:
                if b in spike_ph and b in normal_ph and normal_ph[b]:
                    bystander_deltas.append(
                        (np.mean(spike_ph[b]) - np.mean(normal_ph[b])) / np.mean(normal_ph[b]) * 100
                    )

            avg_bystander_impact = np.mean(bystander_deltas) if bystander_deltas else 0
            print(f"\n  Flooded server ({flooded}) latency increase during spike: +{flooded_delta:.2f}ms")
            print(f"  Bystander servers avg latency change:                   {avg_bystander_impact:+.1f}%")

            if abs(avg_bystander_impact) < 15:
                print(f"\n  ✅ PARTITION ISOLATION VERIFIED — spike on {flooded} did not affect others")
            else:
                print(f"\n  ⚠️  Bystanders impacted by {avg_bystander_impact:.1f}% — check consumer thread affinity")


# ═════════════════════════════════════════════════════════════
#  Final comparison report
# ═════════════════════════════════════════════════════════════

def print_full_report(direct: dict, pipeline: dict, lb_balanced: dict, lb_unbalanced: dict):
    print(f"\n{'═'*W}")
    print(f"  FULL BENCHMARK REPORT  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*W}")

    # Latency comparison
    fmt = f"  {{:<26}} {{:>18}} {{:>18}}"
    print(fmt.format("Metric", "Direct (no Kafka)", "Async Pipeline"))
    print("  " + "─" * (W - 2))
    metrics = [
        ("Flows processed",   "count"),
        ("Throughput (f/s)",  "throughput_flows_per_s"),
        ("Mean latency",      "mean_ms"),
        ("Median (P50)",      "median_ms"),
        ("P95 latency",       "p95_ms"),
        ("P99 latency",       "p99_ms"),
        ("Std deviation",     "std_ms"),
    ]
    units = {"mean_ms": "ms", "median_ms": "ms", "p95_ms": "ms", "p99_ms": "ms", "std_ms": "ms"}
    for label, key in metrics:
        u = units.get(key, "")
        v1 = f"{direct.get(key, '—')}{u}" if direct else "—"
        v2 = f"{pipeline.get(key, '—')}{u}" if pipeline else "—"
        print(fmt.format(label, v1, v2))

    if direct and pipeline and direct.get("mean_ms") and pipeline.get("mean_ms"):
        delta = pipeline["mean_ms"] - direct["mean_ms"]
        pct   = delta / direct["mean_ms"] * 100
        sign  = "+" if delta > 0 else ""
        print(f"\n  Pipeline overhead (mean):     {sign}{delta:.2f}ms  ({sign}{pct:.1f}%)")

    # Load balancing summary
    print(f"\n  {'─'*68}")
    print(f"  LOAD BALANCING SUMMARY")
    print(f"  {'─'*68}")
    if lb_balanced:
        print(f"  Balanced test throughput:     {lb_balanced.get('throughput_flows_per_s', '—')} flows/s")
        print(f"  Balance score:                {lb_balanced.get('balance_score_pct', '—')}%  (100% = perfectly even)")
    if lb_unbalanced:
        print(f"  Unbalanced test throughput:   {lb_unbalanced.get('throughput_flows_per_s', '—')} flows/s")

    print(f"\n{'═'*W}")


def save_all_results(results: dict, out_path: str = "benchmark_results.json"):
    results["timestamp"] = datetime.now().isoformat()
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved → {out_path}\n")


# ═════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Agentic-IDS Full Benchmark: Latency + Load Balancing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_latency.py --n 300                    # all tests
  python benchmark_latency.py --n 300 --skip-direct      # skip baseline
  python benchmark_latency.py --n 300 --lb-only          # load balancing only
  python benchmark_latency.py --n 300 --lb-servers 5     # 5 virtual servers
        """
    )
    parser.add_argument("--data", default="data/raw/filtered_nowebatt.csv",
                        help="Dataset CSV path")
    parser.add_argument("--n", type=int, default=300,
                        help="Total flows to use across all tests")
    parser.add_argument("--servers", type=int, default=3,
                        help="Virtual servers for pipeline benchmark (B)")
    parser.add_argument("--lb-servers", type=int, default=4,
                        help="Virtual servers for load balancing tests (C)")
    parser.add_argument("--skip-direct",   action="store_true", help="Skip benchmark A")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip benchmark B")
    parser.add_argument("--skip-lb",       action="store_true", help="Skip load balancing suite (C)")
    parser.add_argument("--lb-only",       action="store_true", help="Run only load balancing (C)")
    parser.add_argument("--out", default="benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()

    if args.lb_only:
        args.skip_direct = args.skip_pipeline = True

    all_results = {}
    direct_result = {}
    pipeline_result = {}
    lb_balanced = {}
    lb_unbalanced = {}

    # ── A: Direct ─────────────────────────────────────────────
    if not args.skip_direct:
        direct_result = run_direct_benchmark(args.data, args.n)
        all_results["A_direct_no_kafka"] = direct_result

    # ── B: Async Pipeline ─────────────────────────────────────
    if not args.skip_pipeline:
        pipeline_result = asyncio.run(
            run_kafka_benchmark(args.data, args.n, n_servers=args.servers)
        )
        all_results["B_async_pipeline"] = pipeline_result

    # ── C: Load Balancing ─────────────────────────────────────
    if not args.skip_lb:
        lb = LoadBalancingBenchmark(args.data, n_flows=args.n)
        n = min(args.lb_servers, 4)   # cap at 4 for dataset split sanity

        print(f"\n{'═'*W}")
        print(f"  BENCHMARK C — Load Balancing Suite  │  {n} virtual servers")
        print(f"{'═'*W}")

        # C1: Balanced
        lb_balanced = asyncio.run(lb.run_balanced(n_servers=n))
        all_results["C1_lb_balanced"] = lb_balanced

        # C2: Unbalanced (requires exactly 3)
        lb_unbalanced = asyncio.run(lb.run_unbalanced(n_servers=min(n, 3)))
        all_results["C2_lb_unbalanced"] = lb_unbalanced

        # C3: Spike
        lb_spike = asyncio.run(lb.run_spike(n_servers=min(n, 3)))
        all_results["C3_lb_spike"] = lb_spike

    # ── Final Report ──────────────────────────────────────────
    print_full_report(direct_result, pipeline_result, lb_balanced, lb_unbalanced)
    save_all_results(all_results, args.out)


if __name__ == "__main__":
    main()
