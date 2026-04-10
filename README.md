UAV-Assisted Hierarchical Federated Learning with QSGD

A research-grade simulation framework for Hierarchical Federated Learning (HFL) in UAV-assisted IoT networks, integrating gradient compression (Top-K, QSGD) and quorum-based device selection for communication-efficient distributed learning.

📌 What This Project Does

This repo simulates a multi-tier federated learning system:

📱 IoT devices → train locally
🧩 Devices → grouped via clustering
🚁 UAVs → act as intermediate aggregators
🌍 Global server → updates final model

And then compares multiple strategies to answer one question:

How do we reduce communication cost without wrecking accuracy?

🧠 Core Ideas (a.k.a. why this isn’t just another FL repo)
Hierarchical FL (Clustered FL)
Reduces communication overhead by aggregating locally before global updates
Top-K Compression + Error Feedback
Sends only the most important gradients
QSGD Quantization
Compresses gradients into fewer bits while preserving convergence
(based on QSGD paper)
Quorum Selection
Selects only a subset of devices per round while ensuring fairness
Non-IID Data Simulation
Because real-world data is messy and annoying
📂 Project Structure
.
├── main.py              # Runs full experiment pipeline
├── federated.py        # All 6 FL methods (A–F)
├── devices.py          # IoT device simulation
├── clustering.py       # KMeans clustering
├── compression.py      # Top-K + QSGD
├── data_loader.py      # MNIST / CIFAR + partitioning
├── model.py            # CNN model
├── metrics.py          # Metrics + summaries
├── plotting.py         # Graph generation
├── config.py           # Experiment configs
└── results/            # Outputs
🚀 How to Run
1. Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn
2. Run experiment
🧪 Toy Mode (fast, MNIST)
python main.py --mode toy
🔬 Full Mode (CIFAR-10)
python main.py --mode full
🧪 Methods Implemented
Code	Method
A	Standard Federated Learning (FedAvg)
B	Clustered Federated Learning
C	Cluster + Top-K + Error Feedback
D	Cluster + QSGD
E	Cluster + Top-K + Quorum
F	Cluster + QSGD + Quorum

All implemented inside federated.py, because apparently one file needed to carry the entire research paper.

📊 Outputs

After running, you get:

results/
├── toy/ or full/
│   ├── *_history.json
│   ├── accuracy_vs_rounds.png
│   ├── loss_vs_rounds.png
│   ├── latency_vs_rounds.png
│   ├── communication_vs_rounds.png
│   └── tradeoff plots

Metrics include:

Accuracy (best & final)
Training loss
Latency per round
Communication cost (MB)
Active devices
📡 System Modeling
🧩 Device Simulation

Each device has:

Compute power
Bandwidth
Distance (affects latency)
Dataset size
🧠 Clustering

Uses K-Means on:

Distance
Bandwidth
Compute
Network coefficient
📦 Compression
Top-K → sparse updates
QSGD → quantized updates
📚 Datasets
MNIST (toy mode)
CIFAR-10 (full mode)

Supports:

IID split
Non-IID (Dirichlet distribution)
📈 Visualizations

Auto-generated plots:

Accuracy vs Rounds
Loss vs Rounds
Latency vs Rounds
Communication vs Rounds
Accuracy vs Communication trade-off
Accuracy vs Latency trade-off
🔬 Why This Matters

This repo lets you experimentally analyze:

Communication vs accuracy trade-offs
Impact of clustering in FL
Efficiency of gradient compression
Fairness vs performance (quorum selection)
Latency-aware distributed training

Basically, it’s a controlled sandbox for problems people pretend are “solved” in papers.

🧾 Citation
@article{uav_hfl_qsgd_2026,
  title={Hierarchical Federated Learning in UAV-Assisted IoT Networks with Compression},
  author={Pragya Rathal},
  journal={IEEE Transactions (Target)},
  year={2026}
}
⚠️ Notes
Runs on CPU, but GPU helps
Fully reproducible via seeds
If results look weird, it’s probably your config, not the math
