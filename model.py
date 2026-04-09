from dataclasses import dataclass
from typing import List


@dataclass
class DummyCNN:
    input_dim: int
    hidden_dim: int
    output_dim: int
    weights: List[float]

    @classmethod
    def create(cls, input_dim: int = 16, hidden_dim: int = 8, output_dim: int = 10) -> "DummyCNN":
        total = input_dim * hidden_dim + hidden_dim * output_dim
        return cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, weights=[0.0] * total)

    def num_params(self) -> int:
        return len(self.weights)

    def state_vector(self) -> List[float]:
        return list(self.weights)

    def load_state_vector(self, vec: List[float]) -> None:
        self.weights = list(vec)
