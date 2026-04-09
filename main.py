import argparse
from pathlib import Path

import pandas as pd

from config import METHODS, build_config
from data_loader import build_federated_dataloaders, infer_channels
from devices import generate_devices
from federated import run_method
from metrics import aggregate_seed_metrics, summarize_method
from plotting import generate_comparison_plot, generate_method_plots


def pick_best(summary: pd.DataFrame, metric: str, ascending: bool, k: int = 1) -> pd.DataFrame:
    return summary.sort_values(metric, ascending=ascending).head(k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["toy", "full"], required=True)
    parser.add_argument("--iid", action="store_true", help="Use IID splits (default: Dirichlet non-IID)")
    args = parser.parse_args()

    sim_cfg = build_config(args.mode, iid=args.iid)
    mode_cfg = sim_cfg.mode_config
    in_channels = infer_channels(mode_cfg.dataset)

    root_results = Path("results")
    mode_dir = root_results / args.mode
    summaries_dir = root_results / "summaries"
    mode_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    method_summaries = []
    total_plot_count = 0

    for method in METHODS:
        seed_dfs = []
        for seed in mode_cfg.seeds:
            devices = generate_devices(mode_cfg.num_devices, seed)
            train_loaders, test_loader, _ = build_federated_dataloaders(mode_cfg, sim_cfg.iid, seed)
            run_df = run_method(sim_cfg, method, devices, train_loaders, test_loader, in_channels, seed)
            run_df["seed"] = seed
            seed_dfs.append(run_df)

        agg_df = aggregate_seed_metrics(seed_dfs)

        method_dir = mode_dir / method.name
        method_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = method_dir / "metrics.csv"
        agg_df.to_csv(metrics_path, index=False)

        plots_dir = method_dir / "plots"
        total_plot_count += generate_method_plots(agg_df, plots_dir, method.name)

        method_summaries.append(summarize_method(agg_df, method.name, args.mode))

    summary_df = pd.DataFrame(method_summaries)
    mode_summary_path = summaries_dir / f"{args.mode}_summary.csv"
    summary_df.to_csv(mode_summary_path, index=False)

    aggregate_path = summaries_dir / "aggregate.csv"
    if aggregate_path.exists():
        old = pd.read_csv(aggregate_path)
        merged = pd.concat([old[old["mode"] != args.mode], summary_df], ignore_index=True)
    else:
        merged = summary_df.copy()
    merged.to_csv(aggregate_path, index=False)

    generate_comparison_plot(summary_df, summaries_dir / f"{args.mode}_comparison.png")
    total_plot_count += 1

    latency_best = pick_best(summary_df, "avg_latency", ascending=True)
    comm_best = pick_best(summary_df, "total_communication_mb", ascending=True)
    accuracy_best = pick_best(summary_df, "best_accuracy", ascending=False)

    print("\nMethod comparison table")
    print(summary_df.to_string(index=False))

    print("\nBest method under latency constraint:")
    print(latency_best.to_string(index=False))

    print("\nBest method under communication constraint:")
    print(comm_best.to_string(index=False))

    print("\nBest method under accuracy constraint:")
    print(accuracy_best.to_string(index=False))

    print("\nOutput locations:")
    print(f"- Per-method outputs: {mode_dir.resolve()}")
    print(f"- Summary outputs: {summaries_dir.resolve()}")
    print(f"- Aggregate output: {aggregate_path.resolve()}")
    print(f"All plots saved in {root_results.resolve()}/")
    print(f"Total plots generated: {total_plot_count}")


if __name__ == "__main__":
    main()
