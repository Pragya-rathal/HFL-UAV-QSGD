import argparse
import random

import numpy as np
import torch

from config import build_config
from data_loader import build_dataloaders
from devices import Device
from federated import FederatedTrainer
from model import SimpleMLP
from plotting import plot_history


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical Federated Learning Scaffold")
    parser.add_argument("--mode", type=str, default="toy", choices=["toy", "full"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args.mode)
    set_seed(cfg.seed)

    device_loaders, test_loader = build_dataloaders(
        mode=cfg.mode,
        num_devices=cfg.num_devices,
        batch_size=cfg.batch_size,
        input_dim=cfg.input_dim,
        num_classes=cfg.num_classes,
        seed=cfg.seed,
    )

    devices = [
        Device(
            device_id=i,
            lr=cfg.lr,
            bandwidth=10.0 + i,
            compute_power=1.0 + 0.1 * i,
        )
        for i in range(cfg.num_devices)
    ]

    global_model = SimpleMLP(cfg.input_dim, cfg.hidden_dim, cfg.num_classes)

    trainer = FederatedTrainer(
        global_model=global_model,
        devices=devices,
        device_loaders=device_loaders,
        test_loader=test_loader,
        num_clusters=cfg.num_clusters,
        local_epochs=cfg.local_epochs,
        global_rounds=cfg.global_rounds,
        compression_bits=cfg.compression_bits,
    )

    history = trainer.run()
    out_plot = plot_history(history)

    final_loss = history["loss"][-1] if history["loss"] else 0.0
    final_acc = history["accuracy"][-1] if history["accuracy"] else 0.0
    final_lat = history["latency"][-1] if history["latency"] else 0.0

    print(f"mode={cfg.mode}")
    print(f"final_loss={final_loss:.4f}")
    print(f"final_accuracy={final_acc:.4f}")
    print(f"avg_latency={final_lat:.4f}")
    print(f"plot={out_plot}")


if __name__ == "__main__":
    main()
