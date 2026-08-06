import argparse
import ast
import csv
import html
import math
from datetime import datetime
from pathlib import Path


TOP1_KEYS = ("top1", "eval_top1", "eval_top1_ema", "acc1", "valid_top1")
TOP5_KEYS = ("top5", "eval_top5", "eval_top5_ema", "acc5", "valid_top5")
LOSS_KEYS = ("loss", "eval_loss", "valid_loss")


def load_yaml_loose(path):
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        out = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
        return out


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
            items = parsed if isinstance(parsed, list) else text.split()
        except Exception:
            items = text.split()

    out = {}
    for item in items:
        if isinstance(item, str) and "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
    return out


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_value(row, keys, default=""):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def as_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip().replace("%", ""))
    except Exception:
        return None


def last_summary_row(run_dir):
    rows = read_csv_rows(run_dir / "summary.csv")
    return rows[-1] if rows else {}


def sort_bank_name(bank):
    text = Path(str(bank)).stem
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 9999


def classify_run(args, model_kwargs):
    mode = str(model_kwargs.get("pattern_mode", ""))
    bank_path = str(model_kwargs.get("pattern_bank", ""))
    bank_name = Path(bank_path).name
    bank_stem = Path(bank_path).stem
    fixed = str(model_kwargs.get("fixed_pattern", ""))
    budget = first_value(args, ("pattern_budget_target", "pattern-budget-target"), "")

    if mode == "fixed":
        return "Q1", f"fixed_{fixed}", fixed or "fixed"
    if mode == "sampled_supernet" and bank_name == "patterns_6.yml":
        return "Q2", "sampled_supernet_patterns6", "sampled_supernet"
    if mode == "router" and bank_name == "bank4.yml":
        return "Q3A", "router_equal_cost_bank4", "router_bank4"
    if mode == "router" and bank_name == "patterns_mixed.yml":
        suffix = f"budget{budget}" if budget else "budget_unknown"
        return "Q3B", f"router_mixed_{suffix}", suffix
    if bank_name == "original4.yml":
        suffix = f"budget{budget}" if budget else "budget_unknown"
        if mode in ("sampled_uniform", "sampled_quota"):
            return "ReducedBank", f"original4_{mode}", fixed or mode
        if mode == "router":
            return "ReducedBank", f"original4_router_{suffix}", f"router_{suffix}"
        return "ReducedBank", f"original4_{mode}", fixed or mode or "original4"
    if bank_name.startswith("random_budget"):
        suffix = f"budget{budget}" if budget else "budget_unknown"
        if mode in ("sampled_uniform", "sampled_quota"):
            return "RandomBank", f"{bank_stem}_{mode}", fixed or mode
        if mode == "router":
            return "RandomBank", f"{bank_stem}_router_{suffix}", f"router_{suffix}"
        return "RandomBank", f"{bank_stem}_{mode}", fixed or mode or bank_stem
    if bank_name in ("bank2.yml", "bank4.yml", "bank8.yml"):
        return "Q4", f"{bank_stem}_{mode}", f"{bank_stem}_{mode}"
    return "unknown", "_".join(x for x in (mode, bank_stem, fixed) if x) or "unknown", fixed or mode or bank_stem or "unknown"


def collect_runs(output_dir, after=None, before=None):
    rows = []
    for run_dir in sorted(Path(output_dir).glob("pattern_mlp_vit_small_patch16_224_depth12_*")):
        args_path = run_dir / "args.yaml"
        if not args_path.exists():
            continue
        started_at = datetime.fromtimestamp(args_path.stat().st_mtime)
        if after and started_at < after:
            continue
        if before and started_at > before:
            continue

        args = load_yaml_loose(args_path)
        model_kwargs = parse_model_kwargs(args.get("model_kwargs"))
        summary = last_summary_row(run_dir)
        question, experiment, label = classify_run(args, model_kwargs)

        top1 = first_value(summary, TOP1_KEYS)
        top5 = first_value(summary, TOP5_KEYS)
        loss = first_value(summary, LOSS_KEYS)
        epoch = first_value(summary, ("epoch",), "")
        complete = "yes" if top1 != "" else "no"

        row = {
            "question": question,
            "experiment": experiment,
            "label": label,
            "complete": complete,
            "started_at": started_at.isoformat(timespec="seconds"),
            "epoch": epoch,
            "top1": top1,
            "top5": top5,
            "loss": loss,
            "pattern_mode": model_kwargs.get("pattern_mode", ""),
            "pattern_bank": model_kwargs.get("pattern_bank", ""),
            "fixed_pattern": model_kwargs.get("fixed_pattern", ""),
            "pattern_budget_target": first_value(args, ("pattern_budget_target", "pattern-budget-target"), ""),
            "epochs_config": first_value(args, ("epochs",), ""),
            "run_dir": str(run_dir),
        }
        rows.append(row)

        # The same bank4 router configuration is Q3A, but Q4 also needs the
        # bank4/router point to make bank-size plots complete.
        if model_kwargs.get("pattern_mode") == "router" and Path(str(model_kwargs.get("pattern_bank", ""))).name == "bank4.yml":
            dup = dict(row)
            dup["question"] = "Q4"
            dup["experiment"] = "bank4_router"
            dup["label"] = "bank4_router"
            rows.append(dup)
    return rows


def completed(rows):
    out = []
    for row in rows:
        top1 = as_float(row.get("top1"))
        if top1 is not None and math.isfinite(top1):
            new_row = dict(row)
            new_row["_top1"] = top1
            out.append(new_row)
    return out


def try_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def bar_plot(plt, rows, title, path):
    rows = completed(rows)
    if not rows:
        return False
    labels = [r.get("label") or r.get("experiment") for r in rows]
    values = [r["_top1"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(labels)), 4.5))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel("Top-1")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def q4_grouped_plot(plt, rows, path):
    rows = completed(rows)
    if not rows:
        return False
    banks = sorted({Path(str(r.get("pattern_bank", ""))).stem or r.get("label", "") for r in rows}, key=sort_bank_name)
    modes = sorted({r.get("pattern_mode", "") for r in rows})
    values = {}
    for row in rows:
        bank = Path(str(row.get("pattern_bank", ""))).stem or row.get("label", "")
        values[(bank, row.get("pattern_mode", ""))] = row["_top1"]

    width = 0.8 / max(1, len(modes))
    x_positions = list(range(len(banks)))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(banks)), 4.5))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for i, mode in enumerate(modes):
        xs = [x - 0.4 + width / 2 + i * width for x in x_positions]
        ys = [values.get((bank, mode), float("nan")) for bank in banks]
        ax.bar(xs, ys, width=width, label=mode, color=colors[i % len(colors)])
        for xpos, value in zip(xs, ys):
            if math.isfinite(value):
                ax.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(banks)
    ax.set_title("Q4 Pattern Bank Ablation")
    ax.set_ylabel("Top-1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def table_html(title, rows):
    if not rows:
        return f"<h2>{html.escape(title)}</h2><p>No rows.</p>"
    keys = [k for k in rows[0].keys() if not k.startswith("_")]
    parts = [f"<h2>{html.escape(title)}</h2>", "<table><thead><tr>"]
    parts.extend(f"<th>{html.escape(k)}</th>" for k in keys)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{html.escape(str(row.get(k, '')))}</td>" for k in keys)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def make_takeaways(rows):
    lines = []
    done = completed(rows)
    if not done:
        return ["No completed runs with top1 were found."]

    real_done = [r for r in done if not (r["question"] == "Q4" and r["experiment"] == "bank4_router" and "Q3A" in [x["question"] for x in done])]
    lines.append(f"Completed runs with top1: {len(real_done)}.")

    for question in ("Q1", "Q2", "Q3A", "Q3B", "Q4"):
        sub = [r for r in done if r["question"] == question]
        if not sub:
            continue
        best = max(sub, key=lambda r: r["_top1"])
        lines.append(f"{question} best: {best['label']} ({best['_top1']:.3f} top1, epoch {best.get('epoch', '')}).")

    q1 = sorted([r for r in done if r["question"] == "Q1"], key=lambda r: r["_top1"], reverse=True)
    if len(q1) >= 2:
        lines.append(f"Q1 spread: {q1[0]['_top1'] - q1[-1]['_top1']:.3f} top1 points.")

    q3a = [r for r in done if r["question"] == "Q3A"]
    q1_best = max(q1, key=lambda r: r["_top1"]) if q1 else None
    if q3a and q1_best:
        delta = max(q3a, key=lambda r: r["_top1"])["_top1"] - q1_best["_top1"]
        lines.append(f"Q3A vs best Q1 fixed: {delta:+.3f} top1 points.")

    incomplete = [r for r in rows if r.get("complete") != "yes"]
    if incomplete:
        lines.append(f"Incomplete/no-summary runs: {len(incomplete)}. They are listed in all_runs.csv but excluded from plots.")
    return lines


def parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(value)


def main():
    parser = argparse.ArgumentParser(description="Analyze all PatternMLP output/train runs and create visual reports.")
    parser.add_argument("--output-dir", default="output/train")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--after", help="Only include runs started after this local ISO time.")
    parser.add_argument("--before", help="Only include runs started before this local ISO time.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_runs(args.output_dir, after=parse_time(args.after), before=parse_time(args.before))
    rows_sorted = sorted(rows, key=lambda r: (r["question"], r["experiment"], r["started_at"]))
    done_sorted = sorted(completed(rows), key=lambda r: (r["question"], r["experiment"], -r["_top1"]))

    write_csv(out_dir / "all_runs.csv", rows_sorted)
    write_csv(out_dir / "completed_runs_ranked.csv", done_sorted)
    for question in ("Q1", "Q2", "Q3A", "Q3B", "Q4", "unknown"):
        sub = [r for r in done_sorted if r["question"] == question]
        if sub:
            write_csv(out_dir / f"{question.lower()}_ranked.csv", sub)

    plt = try_matplotlib()
    images = []
    if plt:
        plot_specs = [
            ("all_top1.png", done_sorted, "All Completed PatternMLP Runs"),
            ("q1_fixed_top1.png", [r for r in done_sorted if r["question"] == "Q1"], "Q1 Fixed Pattern Ablation"),
            ("q3_router_top1.png", [r for r in done_sorted if r["question"] in ("Q3A", "Q3B")], "Q3 Router Experiments"),
        ]
        for filename, sub_rows, title in plot_specs:
            path = out_dir / filename
            if bar_plot(plt, sub_rows, title, path):
                images.append(filename)
        q4_path = out_dir / "q4_bank_top1.png"
        if q4_grouped_plot(plt, [r for r in done_sorted if r["question"] == "Q4"], q4_path):
            images.append(q4_path.name)

    takeaways = make_takeaways(rows)
    md = [
        "# PatternMLP Experiment Analysis",
        "",
        "## Takeaways",
        *[f"- {line}" for line in takeaways],
        "",
        "## Files",
        "- all_runs.csv",
        "- completed_runs_ranked.csv",
        *[f"- {name}" for name in images],
    ]
    if not plt:
        md.append("- PNG plots skipped because matplotlib is unavailable.")
    (out_dir / "analysis_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>PatternMLP Experiment Analysis</title>",
        "<style>body{font-family:Arial,sans-serif;margin:28px;line-height:1.45}table{border-collapse:collapse;width:100%;font-size:12px;margin:14px 0 28px}th,td{border:1px solid #ddd;padding:5px 7px;text-align:left}th{background:#f4f6f8;position:sticky;top:0}img{max-width:1100px;width:100%;border:1px solid #ddd;margin:10px 0 24px}.note{background:#f7f7f7;padding:10px 12px;border-left:4px solid #777}</style>",
        "</head><body>",
        "<h1>PatternMLP Experiment Analysis</h1>",
        "<div class='note'>This report is built from output/train/*/args.yaml and summary.csv. Manifest CSV files are not required.</div>",
        "<h2>Takeaways</h2><ul>",
        *[f"<li>{html.escape(line)}</li>" for line in takeaways],
        "</ul>",
        *[f"<h2>{html.escape(name)}</h2><img src='{html.escape(name)}'>" for name in images],
        table_html("Completed Runs Ranked", done_sorted),
        table_html("All Runs", rows_sorted),
        "</body></html>",
    ]
    (out_dir / "analysis_dashboard.html").write_text("\n".join(html_parts), encoding="utf-8")

    print(f"Wrote analysis to {out_dir}")
    print(f"- {out_dir / 'analysis_summary.md'}")
    print(f"- {out_dir / 'analysis_dashboard.html'}")
    print(f"- {out_dir / 'all_runs.csv'}")
    print(f"- {out_dir / 'completed_runs_ranked.csv'}")


if __name__ == "__main__":
    main()
