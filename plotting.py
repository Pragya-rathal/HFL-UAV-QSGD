from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history: Dict[str, List[float]], out_path: str = "training_curve.png") -> str:
    rounds = list(range(1, len(history.get("loss", [])) + 1))

    plt.figure(figsize=(6, 4))
    if "loss" in history:
        plt.plot(rounds, history["loss"], label="loss")
    if "accuracy" in history:
        plt.plot(rounds, history["accuracy"], label="accuracy")
    plt.xlabel("Round")
    plt.ylabel("Metric")
    plt.title("Federated Training Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
