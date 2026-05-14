from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from pathlib import Path

from microalpha_exec_lab.alpha_cross_asset_benchmark import AlphaPhase4Config, run_alpha_phase4


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    cfg = AlphaPhase4Config(
        optiver_root=repo_root / "data" / "optiver",
        binance_root=repo_root / "data" / "binance",
        output_dir=repo_root / "reports",
    )
    out = run_alpha_phase4(cfg)
    print("=== Phase 4 summary ===")
    print(out["summary"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nArtifacts written to:", repo_root / "reports")
