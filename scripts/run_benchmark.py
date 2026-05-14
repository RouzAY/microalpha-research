from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from microalpha_exec_lab.benchmark import BenchmarkConfig, run_full_benchmark


def main() -> None:
    cfg = BenchmarkConfig(
        data_root=ROOT / "data" / "optiver",
        output_dir=ROOT / "reports",
        symbol="AAPL",
    )
    outputs = run_full_benchmark(cfg)
    real_summary = outputs["real_summary"]
    synth_summary = outputs["synthetic_summary"]

    print("\n=== Held-out real data (mean IS bps) ===")
    print(real_summary.to_string(index=False))
    print("\n=== Synthetic stress test (mean IS bps) ===")
    print(synth_summary.to_string(index=False))
    print(f"\nReports written to: {ROOT / 'reports'}")


if __name__ == "__main__":
    main()
