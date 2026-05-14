from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BootstrapGenerator:
    train_frames: list[pd.DataFrame]
    random_seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def make_episode(self, block_size: int = 60, episode_len: int = 900) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        while sum(len(piece) for piece in pieces) < episode_len:
            frame = self.train_frames[int(self.rng.integers(0, len(self.train_frames)))]
            start = int(self.rng.integers(0, len(frame) - block_size - 1))
            pieces.append(frame.iloc[start : start + block_size].copy())
        out = pd.concat(pieces, ignore_index=True).iloc[:episode_len].copy()
        out["ts_ns"] = np.arange(len(out), dtype=np.int64)
        return out

    def make_many(self, n_episodes: int = 50, block_size: int = 60, episode_len: int = 900) -> list[pd.DataFrame]:
        return [self.make_episode(block_size=block_size, episode_len=episode_len) for _ in range(n_episodes)]
