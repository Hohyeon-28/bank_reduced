import argparse
import csv
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Q4 pattern bank size ablation launcher.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--banks", nargs="+", default=["bank2", "bank4", "bank8"])
    parser.add_argument("--output-csv", default="results/pattern_bank_ablation.csv")
    parser.add_argument("--epochs", default="600")
    parser.add_argument("--batch-size", default="512")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--execute", action="store_true")
    args, extra = parser.parse_known_args()

    rows = []
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    for bank in args.banks:
        bank_path = f"configs/pattern_banks/{bank}.yml"
        for mode in ("sampled_supernet", "router"):
            cmd = [
                "torchrun", "--nproc_per_node=2", "train_sh.py", args.data_dir,
                "--model", "pattern_mlp_vit_small_patch16_224_depth12",
                "--epochs", args.epochs,
                "--batch-size", args.batch_size,
                "--seed", args.seed,
                "--model-kwargs", f"pattern_mode={mode}", f"pattern_bank={bank_path}",
                *extra,
            ]
            rows.append({
                "bank": bank,
                "mode": mode,
                "command": " ".join(cmd),
                "top1": "",
                "pattern_usage_entropy": "",
                "oracle_avg_active_mlp": "",
                "latency_ms": "",
            })
            if args.execute:
                subprocess.run(cmd, check=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
