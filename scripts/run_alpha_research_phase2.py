from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from microalpha_exec_lab.alpha_research import AlphaBenchmarkConfig, run_alpha_phase2


if __name__ == "__main__":
    cfg = AlphaBenchmarkConfig(
        data_root=ROOT / "data" / "optiver",
        symbol="AAPL",
        output_dir=ROOT / "reports",
    )
    out = run_alpha_phase2(cfg)
    print(out["summary"])
