from pathlib import Path
import sys

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from microalpha_exec_lab.alpha_real_benchmark import AlphaPhase3Config, run_alpha_phase3
from microalpha_exec_lab.alpha_real_models import RealStrategyConfig

if __name__ == "__main__":
    cfg = AlphaPhase3Config(
        data_root=ROOT / "data" / "optiver",
        symbol="AAPL",
        output_dir=ROOT / "reports",
        strategy=RealStrategyConfig(
            horizon=30,
            warmup=800,
            hold_steps=30,
            threshold_quantile=0.85,
            cost_coeff=0.12,
            ridge_lambda=3000.0,
            beta_switch_threshold=2.5,
            micro_weight=0.7,
            ml_weight=0.3,
        ),
        eval_start_index=1,
        heldout_days=2,
    )
    out = run_alpha_phase3(cfg)
    print(out["summary"])
    print("\nHeld-out block:\n")
    print(out["heldout_summary"])
