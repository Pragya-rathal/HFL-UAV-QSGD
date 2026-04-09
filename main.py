import argparse
import csv
import random
from pathlib import Path

from config import build_config
from data_loader import build_federated_datasets
from devices import generate_devices
from federated import run_method
from metrics import summarize_metrics, write_metrics_csv
from plotting import generate_plots


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["mode", "method", "best_accuracy", "avg_latency", "total_communication_mb", "convergence_round"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["toy", "full"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iid", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    cfg = build_config(seed=args.seed)
    mode_cfg = cfg.modes[args.mode]

    # Build once to ensure imports and data plumbing are exercised.
    build_federated_datasets(mode_cfg=mode_cfg, seed=args.seed, iid=args.iid)

    results_root = Path("results")
    mode_root = results_root / args.mode
    summary_root = results_root / "summaries"

    summaries: list[dict] = []
    total_plots = 0

    for method in cfg.methods:
        devices = generate_devices(mode_cfg.num_devices, seed=args.seed)
        round_metrics = run_method(mode_cfg=mode_cfg, method=method, devices=devices, seed=args.seed)

        method_dir = mode_root / method
        metrics_path = method_dir / "metrics.csv"
        plots_dir = method_dir / "plots"

        write_metrics_csv(metrics_path, round_metrics)
        total_plots += generate_plots(round_metrics, plots_dir)

        summaries.append(summarize_metrics(round_metrics, method=method, mode=args.mode))

    mode_summary = summary_root / f"{args.mode}_summary.csv"
    aggregate_summary = summary_root / "aggregate.csv"

    _write_summary_csv(mode_summary, summaries)
    _write_summary_csv(aggregate_summary, summaries)

    print("Method comparison table")
    for row in summaries:
        print(row)

    best_latency = min(summaries, key=lambda x: x["avg_latency"]) if summaries else {}
    best_comm = min(summaries, key=lambda x: x["total_communication_mb"]) if summaries else {}
    best_acc = max(summaries, key=lambda x: x["best_accuracy"]) if summaries else {}

    print("Best method under latency constraint:", best_latency)
    print("Best method under communication constraint:", best_comm)
    print("Best method under accuracy constraint:", best_acc)

    print("Output locations:")
    print("- Per-method outputs:", str(mode_root.resolve()))
    print("- Summary outputs:", str(summary_root.resolve()))
    print("All plots saved in results/plots/")
    print("Total plots generated:", total_plots)


if __name__ == "__main__":
    main()
