from pathlib import Path
from typing import Iterable

from metrics import RoundMetric


def _write_text_plot(path: Path, title: str, values: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(title + "\n")
        for idx, value in enumerate(values, start=1):
            bars = "#" * int(max(1, value * 20))
            f.write(f"{idx:03d} {value:.4f} {bars}\n")


def generate_plots(metrics: list[RoundMetric], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = [m.accuracy for m in metrics]
    loss = [m.loss for m in metrics]
    lat = [m.latency for m in metrics]
    comm = [m.communication_mb for m in metrics]

    files = [
        ("accuracy_vs_rounds.txt", "Accuracy vs Rounds", acc),
        ("loss_vs_rounds.txt", "Loss vs Rounds", loss),
        ("latency_vs_rounds.txt", "Latency vs Rounds", lat),
        ("communication_vs_rounds.txt", "Communication vs Rounds", comm),
    ]

    for filename, title, values in files:
        _write_text_plot(out_dir / filename, title, values)

    return len(files)
