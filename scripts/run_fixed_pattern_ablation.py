import argparse
import csv
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Q1 fixed pattern placement ablation launcher.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--bank", default="configs/pattern_banks/patterns_6.yml")
    parser.add_argument("--patterns", nargs="+", default=["early6", "mid6", "late6", "alternating6"])
    parser.add_argument("--output-csv", default="results/fixed_pattern_ablation.csv")
    parser.add_argument("--epochs", default="300")
    parser.add_argument("--batch-size", default="512")
    parser.add_argument("--cuda", default="0,1")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--execute", action="store_true")
    args, extra = parser.parse_known_args()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for pattern in args.patterns:
        cmd = [
            "torchrun", "--nproc_per_node=2", "train_sh.py", args.data_dir,
            "--model", "pattern_mlp_vit_small_patch16_224_depth12",
            "--epochs", args.epochs,
            "--batch-size", args.batch_size,
            "--seed", args.seed,
            "--cuda", args.cuda,
            "--model-kwargs", "pattern_mode=fixed", f"pattern_bank={args.bank}", f"fixed_pattern={pattern}",
            *extra,
        ]
        rows.append({
            "model_name": "pattern_mlp_vit_small_patch16_224_depth12",
            "pattern_name": pattern,
            "seed": args.seed,
            "command": " ".join(cmd),
            "checkpoint_path": "",
            "top1": "",
            "top5": "",
            "nll": "",
            "active_mlp_count": "",
            "active_mlp_ratio": "",
            "estimated_mlp_flops_ratio": "",
            "latency_ms_batch1": "",
            "latency_ms_batchN": "",
            "num_parameters": "",
        })
        if args.execute:
            subprocess.run(cmd, check=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
