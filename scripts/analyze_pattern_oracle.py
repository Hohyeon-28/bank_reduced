import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, load_checkpoint
from timm.utils import accuracy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models.early_pattern_vit
from models.early_pattern_vit import load_pattern_bank


def main():
    parser = argparse.ArgumentParser(description="Q2 per-sample oracle pattern analysis.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--bank", default="configs/pattern_banks/patterns_6.yml")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-csv", default="results/pattern_oracle_samples.csv")
    parser.add_argument("--summary-csv", default="results/pattern_oracle_summary.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, patterns = load_pattern_bank(args.bank)
    dataset = create_dataset("", root=args.data_dir, split=args.split, is_training=False)
    model0 = create_model(
        "pattern_mlp_vit_small_patch16_224_depth12",
        pattern_mode="fixed",
        pattern_bank=args.bank,
        fixed_pattern=names[0],
    )
    data_config = resolve_data_config({}, model=model0)
    loader = create_loader(dataset, input_size=data_config["input_size"], batch_size=args.batch_size,
                           is_training=False, use_prefetcher=False, interpolation=data_config["interpolation"],
                           mean=data_config["mean"], std=data_config["std"], num_workers=args.workers)

    models = {}
    for name in names:
        model = create_model(
            "pattern_mlp_vit_small_patch16_224_depth12",
            pattern_mode="fixed",
            pattern_bank=args.bank,
            fixed_pattern=name,
        ).to(device).eval()
        if args.checkpoint:
            load_checkpoint(model, args.checkpoint)
        models[name] = model

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    fixed_correct = Counter()
    selected = defaultdict(Counter)
    sample_index = 0
    with torch.no_grad():
        for input, target in loader:
            input, target = input.to(device), target.to(device)
            per_pattern = []
            for name, model in models.items():
                logits = model(input)
                prob = logits.softmax(dim=1)
                ce = F.cross_entropy(logits, target, reduction="none")
                pred = logits.argmax(dim=1)
                correct = pred.eq(target)
                fixed_correct[name] += int(correct.sum().item())
                per_pattern.append((name, logits, prob, ce, pred, correct))

            for b in range(input.shape[0]):
                choices = []
                for name, logits, prob, ce, pred, correct in per_pattern:
                    count = int(patterns[names.index(name)].sum().item())
                    choices.append({
                        "pattern_name": name,
                        "prediction": int(pred[b].item()),
                        "correct": int(correct[b].item()),
                        "true_class_probability": float(prob[b, target[b]].item()),
                        "cross_entropy": float(ce[b].item()),
                        "top1_confidence": float(prob[b].max().item()),
                        "active_mlp_count": count,
                    })
                min_ce = min(choices, key=lambda r: r["cross_entropy"])
                max_prob = max(choices, key=lambda r: r["true_class_probability"])
                correct_choices = [r for r in choices if r["correct"]]
                cheapest = min(correct_choices, key=lambda r: r["active_mlp_count"]) if correct_choices else min_ce
                oracle_map = {
                    "oracle_min_ce": min_ce,
                    "oracle_max_true_prob": max_prob,
                    "oracle_cheapest_correct": cheapest,
                }
                for oracle_type, row in oracle_map.items():
                    selected[oracle_type][row["pattern_name"]] += 1
                    sample_rows.append({
                        "sample_index": sample_index,
                        "label": int(target[b].item()),
                        "oracle_type": oracle_type,
                        **row,
                    })
                sample_index += 1
                if args.max_samples and sample_index >= args.max_samples:
                    break
            if args.max_samples and sample_index >= args.max_samples:
                break

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    best_fixed_name, best_fixed_correct = max(fixed_correct.items(), key=lambda kv: kv[1])
    summary_rows = []
    for oracle_type, counts in selected.items():
        for name, count in counts.items():
            summary_rows.append({
                "oracle_type": oracle_type,
                "pattern_name": name,
                "selected_count": count,
                "selected_ratio": count / max(sample_index, 1),
                "oracle_top1": "",
                "oracle_avg_active_mlp": "",
                "oracle_avg_active_mlp_ratio": "",
                "fixed_best_pattern_top1": best_fixed_correct / max(sample_index, 1) * 100.0,
                "fixed_best_pattern": best_fixed_name,
                "oracle_gain_over_best_fixed": "",
            })
    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
