import argparse
import ast
import csv
import json
from datetime import datetime
from pathlib import Path


def load_yaml(path):
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        data = {}
        for line in path.read_text().splitlines():
            if ":" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
        return data


def parse_model_kwargs(raw):
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        items = raw
    else:
        text = str(raw).strip()
        try:
            parsed = ast.literal_eval(text)
            items = parsed if isinstance(parsed, list) else [text]
        except Exception:
            items = text.split()

    out = {}
    for item in items:
        if not isinstance(item, str) or "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key] = value
    return out


def read_last_summary(summary_path):
    if not summary_path.exists():
        return {}
    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return rows[-1]


def as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def classify_run(args, model_kwargs):
    mode = model_kwargs.get("pattern_mode", "")
    bank = Path(model_kwargs.get("pattern_bank", "")).name
    fixed = model_kwargs.get("fixed_pattern", "")
    target = args.get("pattern_budget_target", args.get("pattern-budget-target", 0))
    target_f = as_float(target) or 0.0

    if mode == "fixed":
        return "Q1", f"fixed_{fixed}", fixed
    if mode == "sampled_supernet" and bank == "patterns_6.yml":
        return "Q2", "sampled_supernet_patterns6", "sampled_supernet"
    if mode == "router" and bank == "bank4.yml":
        return "Q3A", "router_equal_cost_bank4", "router_bank4"
    if mode == "router" and bank == "patterns_mixed.yml":
        return "Q3B", f"router_mixed_budget{target_f:g}", f"budget{target_f:g}"
    if bank in ("bank2.yml", "bank4.yml", "bank8.yml"):
        return "Q4", f"{Path(bank).stem}_{mode}", f"{Path(bank).stem}_{mode}"
    return "unknown", f"{mode}_{bank}_{fixed}".strip("_"), fixed or mode or bank or "unknown"


def collect_runs(output_dir, router_valid_after=None):
    rows = []
    for run_dir in sorted(Path(output_dir).glob("pattern_mlp_vit_small_patch16_224_depth12_*")):
        args_path = run_dir / "args.yaml"
        if not args_path.exists():
            continue
        args = load_yaml(args_path)
        model_kwargs = parse_model_kwargs(args.get("model_kwargs"))
        started_at = datetime.fromtimestamp(args_path.stat().st_mtime)
        if (
                router_valid_after is not None
                and model_kwargs.get("pattern_mode") == "router"
                and started_at < router_valid_after):
            continue
        q, experiment, label = classify_run(args, model_kwargs)
        summary = read_last_summary(run_dir / "summary.csv")
        top1 = summary.get("eval_top1") or summary.get("top1") or summary.get("eval_top1_ema")
        top5 = summary.get("eval_top5") or summary.get("top5") or summary.get("eval_top5_ema")
        loss = summary.get("eval_loss") or summary.get("loss")
        epoch = summary.get("epoch", "")

        rows.append({
            "question": q,
            "experiment": experiment,
            "label": label,
            "run_dir": str(run_dir),
            "started_at": started_at.isoformat(timespec="seconds"),
            "epoch": epoch,
            "top1": top1,
            "top5": top5,
            "loss": loss,
            "pattern_mode": model_kwargs.get("pattern_mode", ""),
            "pattern_bank": model_kwargs.get("pattern_bank", ""),
            "fixed_pattern": model_kwargs.get("fixed_pattern", ""),
            "pattern_budget_target": args.get("pattern_budget_target", ""),
            "epochs_config": args.get("epochs", ""),
            "model_kwargs": " ".join(f"{k}={v}" for k, v in model_kwargs.items()),
        })
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skipped plots: {exc}")
        return

    def save_bar(sub_rows, title, filename, xlabel="experiment"):
        sub_rows = [r for r in sub_rows if as_float(r.get("top1")) is not None]
        if not sub_rows:
            return
        labels = [r["label"] or r["experiment"] for r in sub_rows]
        values = [as_float(r["top1"]) for r in sub_rows]
        plt.figure(figsize=(max(7, len(labels) * 1.2), 4.5))
        plt.bar(labels, values)
        plt.title(title)
        plt.ylabel("Top-1")
        plt.xlabel(xlabel)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=180)
        plt.close()

    save_bar(rows, "All Pattern MLP Experiments", "all_experiments_top1.png")
    save_bar([r for r in rows if r["question"] == "Q1"], "Q1 Fixed 6-MLP Placement", "q1_fixed_patterns_top1.png", "pattern")
    save_bar([r for r in rows if r["question"] in ("Q3A", "Q3B")], "Q3 Router Experiments", "q3_router_top1.png")
    save_bar([r for r in rows if r["question"] == "Q4"], "Q4 Pattern Bank Ablation", "q4_bank_ablation_top1.png")


def main():
    parser = argparse.ArgumentParser(description="Index and visualize PatternMLP experiment outputs.")
    parser.add_argument("--output-dir", default="output/train")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--router-valid-after",
        help="Exclude router runs started before this local time (ISO format, e.g. 2026-06-29T14:30:00).",
    )
    args = parser.parse_args()

    router_valid_after = datetime.fromisoformat(args.router_valid_after) if args.router_valid_after else None
    rows = collect_runs(args.output_dir, router_valid_after=router_valid_after)
    results_dir = Path(args.results_dir)
    write_csv(results_dir / "run_index.csv", rows)

    summary_rows = sorted(
        rows,
        key=lambda r: (r["question"], r["experiment"], -(as_float(r.get("top1")) or -1.0)),
    )
    write_csv(results_dir / "experiment_summary.csv", summary_rows)
    plot(summary_rows, results_dir / "plots")

    counts = {}
    for row in rows:
        counts[row["question"]] = counts.get(row["question"], 0) + 1
    print(json.dumps({
        "num_runs": len(rows),
        "counts": counts,
        "run_index": str(results_dir / "run_index.csv"),
        "summary": str(results_dir / "experiment_summary.csv"),
        "plots": str(results_dir / "plots"),
    }, indent=2))


if __name__ == "__main__":
    main()
