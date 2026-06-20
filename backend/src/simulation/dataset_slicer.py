"""
Dataset Slicer — splits the existing CSV/XLSX dataset into N chunks,
one per simulated geo-server. Each chunk is then replayed by a FlowProducer.

Industry pattern: "shadow traffic replay" — split by stratified sampling so
each server slice has the same attack/benign ratio as the original dataset.
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.kafka.registry import ServerRegistry

logger = logging.getLogger(__name__)


@dataclass
class ServerSlice:
    """One server's portion of the dataset."""
    server_id: str
    geo_region: str
    geo_label: str
    lat: float
    lon: float
    df: pd.DataFrame          # The actual rows assigned to this server
    slice_index: int          # 0-based index among all slices
    total_slices: int


class DatasetSlicer:
    """
    Splits a dataset across N simulated geo-servers.

    Slicing strategy (selectable):
      - 'stratified'  (default): Each server gets a proportional mix of
                                  BENIGN/DDoS/PortScan — most realistic.
      - 'sequential': Rows split sequentially (first N/k rows → server 0, etc.)
      - 'random':     Random shuffle then split.

    Example:
        slicer = DatasetSlicer(data_path="data/raw/filtered_nowebatt.csv", n_servers=3)
        slices = slicer.slice()
        for s in slices:
            print(f"{s.server_id}: {len(s.df)} rows")
    """

    def __init__(
        self,
        data_path: str,
        n_servers: int,
        strategy: str = "stratified",
        label_column: str = "Label",
        random_seed: int = 42,
    ):
        self.data_path = data_path
        self.n_servers = n_servers
        self.strategy = strategy
        self.label_column = label_column
        self.random_seed = random_seed
        self._df: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Load the dataset (CSV or Excel)."""
        if self._df is not None:
            return self._df

        logger.info(f"[Slicer] Loading dataset: {self.data_path}")
        if self.data_path.endswith(".csv"):
            self._df = pd.read_csv(self.data_path)
        else:
            self._df = pd.read_excel(self.data_path)
        logger.info(f"[Slicer] Loaded {len(self._df)} rows, {len(self._df.columns)} columns")
        return self._df

    def slice(self, server_configs: Optional[List[dict]] = None) -> List[ServerSlice]:
        """
        Split the dataset into N server slices.

        Args:
            server_configs: Optional list of dicts with server metadata.
                            If None, picks from GEO_PRESETS automatically.
        Returns:
            List of ServerSlice objects, one per server.
        """
        df = self.load()

        if server_configs is None:
            presets = ServerRegistry.GEO_PRESETS[: self.n_servers]
            server_configs = [
                {
                    "server_id": f"{p['id']}-{str(i+1).zfill(2)}",
                    "geo_region": p["region"],
                    "geo_label": p["label"],
                    "lat": p["lat"],
                    "lon": p["lon"],
                }
                for i, p in enumerate(presets)
            ]

        chunks = self._split(df)
        slices = []
        for i, (cfg, chunk) in enumerate(zip(server_configs, chunks)):
            slices.append(ServerSlice(
                server_id=cfg["server_id"],
                geo_region=cfg["geo_region"],
                geo_label=cfg["geo_label"],
                lat=cfg["lat"],
                lon=cfg["lon"],
                df=chunk.reset_index(drop=True),
                slice_index=i,
                total_slices=self.n_servers,
            ))
            logger.info(
                f"[Slicer] {cfg['server_id']} ({cfg['geo_label']}): "
                f"{len(chunk)} rows"
                + (
                    f"  labels: {chunk[self.label_column].value_counts().to_dict()}"
                    if self.label_column in chunk.columns
                    else ""
                )
            )
        return slices

    # ─────────────────────────────────────────────────────────
    #  Split strategies
    # ─────────────────────────────────────────────────────────

    def _split(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        if self.strategy == "stratified":
            return self._stratified_split(df)
        elif self.strategy == "sequential":
            return self._sequential_split(df)
        elif self.strategy == "random":
            return self._random_split(df)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def _stratified_split(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Stratified split preserving label distribution per server.
        This is the industry-standard approach — ensures each geo node
        sees a representative mix of traffic types.
        """
        if self.label_column not in df.columns:
            logger.warning(
                f"[Slicer] Label column '{self.label_column}' not found. "
                "Falling back to sequential split."
            )
            return self._sequential_split(df)

        chunks: List[List[pd.DataFrame]] = [[] for _ in range(self.n_servers)]
        for label, group in df.groupby(self.label_column):
            group_shuffled = group.sample(frac=1, random_state=self.random_seed)
            sub_chunks = np.array_split(group_shuffled, self.n_servers)
            for i, sub in enumerate(sub_chunks):
                chunks[i].append(sub)

        return [pd.concat(c, ignore_index=True).sample(frac=1, random_state=self.random_seed)
                for c in chunks]

    def _sequential_split(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        return [chunk for chunk in np.array_split(df, self.n_servers) if len(chunk) > 0]

    def _random_split(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        shuffled = df.sample(frac=1, random_state=self.random_seed)
        return [chunk for chunk in np.array_split(shuffled, self.n_servers) if len(chunk) > 0]

    # ─────────────────────────────────────────────────────────
    #  Convenience: save slices to disk (optional)
    # ─────────────────────────────────────────────────────────

    def save_slices(self, output_dir: str, slices: List[ServerSlice]):
        os.makedirs(output_dir, exist_ok=True)
        for s in slices:
            path = os.path.join(output_dir, f"{s.server_id}.csv")
            s.df.to_csv(path, index=False)
            logger.info(f"[Slicer] Saved {len(s.df)} rows → {path}")
