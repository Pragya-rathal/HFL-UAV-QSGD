# UAV-Assisted Hierarchical Federated Learning (QSGD + Quorum)

## Run

```bash
python main.py --mode toy
python main.py --mode full
```

Outputs are saved under:
- `results/toy/<method>/metrics.csv`
- `results/full/<method>/metrics.csv`
- `results/summaries/*.csv`
- `results/plots/**.png`

## Methods
- A: Standard FL
- B: Clustered FL (no compression)
- C: Clustered FL + Top-K + Error Feedback
- D: Clustered FL + QSGD
- E: Clustered FL + Top-K + Error Feedback + Quorum
- F: Clustered FL + QSGD + Quorum
