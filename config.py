from dataclasses import dataclass


@dataclass
class Config:
    mode: str = "toy"
    seed: int = 42
    input_dim: int = 20
    hidden_dim: int = 32
    num_classes: int = 2
    num_devices: int = 6
    num_clusters: int = 2
    local_epochs: int = 1
    global_rounds: int = 3
    batch_size: int = 16
    lr: float = 0.01
    compression_bits: int = 8


def build_config(mode: str) -> Config:
    if mode == "toy":
        return Config(
            mode="toy",
            seed=42,
            input_dim=20,
            hidden_dim=32,
            num_classes=2,
            num_devices=4,
            num_clusters=2,
            local_epochs=1,
            global_rounds=2,
            batch_size=16,
            lr=0.02,
            compression_bits=8,
        )

    return Config(
        mode="full",
        seed=123,
        input_dim=40,
        hidden_dim=64,
        num_classes=4,
        num_devices=8,
        num_clusters=4,
        local_epochs=2,
        global_rounds=3,
        batch_size=32,
        lr=0.01,
        compression_bits=8,
    )
