🚁 UAV-Assisted Hierarchical Federated Learning with QSGD

A research-grade simulation framework for Hierarchical Federated Learning (HFL) in UAV-assisted IoT networks. This project integrates gradient compression (Top-K, QSGD) and quorum-based device selection to study communication-efficient distributed learning.

📌 Overview

This project simulates a multi-tier federated learning system:

📱 IoT devices perform local training
🧩 Devices are grouped via clustering
🚁 UAVs act as intermediate aggregators
🌍 A global model is updated iteratively

The framework evaluates trade-offs between:

Model accuracy
Communication overhead
Training latency
Device participation fairness
🧠 Key Features
Standard Federated Learning (FedAvg)
Hierarchical (Clustered) Federated Learning
Top-K Gradient Compression with Error Feedback
QSGD Quantization
Quorum-Based Device Selection
IID and Non-IID (Dirichlet) data partitioning
Realistic IoT device simulation
End-to-end experiment pipeline with plots
📂 Project Structure
.
├── main.py              # Entry point
├── config.py            # Experiment configuration
├── federated.py         # FL algorithms (Methods A–F)
├── devices.py           # IoT device simulation
├── clustering.py        # Device clustering (K-Means)
├── compression.py       # Top-K and QSGD
├── data_loader.py       # Dataset loading and partitioning
├── model.py             # CNN model
├── metrics.py           # Metrics computation
├── plotting.py          # Visualization
└── results/             # Output directory
⚙️ Installation
git clone https://github.com/Pragya-rathal/uav-project-QSGD.git
cd uav-project-QSGD

pip install torch torchvision numpy matplotlib scikit-learn
🚀 Running Experiments
Toy Mode (MNIST)
python main.py --mode toy
Full Mode (CIFAR-10)
python main.py --mode full
🧪 Methods Implemented
Method	Description
A	Standard Federated Learning (FedAvg)
B	Clustered Federated Learning
C	Cluster + Top-K + Error Feedback
D	Cluster + QSGD
E	Cluster + Top-K + Quorum Selection
F	Cluster + QSGD + Quorum Selection
📊 Outputs

Results are saved in:

results/
├── toy/ or full/
│   ├── *_history.json
│   ├── accuracy_vs_rounds.png
│   ├── loss_vs_rounds.png
│   ├── latency_vs_rounds.png
│   ├── communication_vs_rounds.png
│   ├── tradeoff_acc_comm.png
│   ├── tradeoff_acc_lat.png
│   └── active_devices.png
Metrics
Best and final accuracy
Training loss
Average and total latency
Communication cost (MB)
Active devices per round
📡 System Model
IoT Devices

Each simulated device includes:

Compute power
Bandwidth
Distance (affects communication latency)
Local dataset size
Clustering

Devices are grouped using K-Means based on:

Distance
Bandwidth
Compute power
Network clustering coefficient
Compression Techniques

Top-K + Error Feedback

Selects top-magnitude gradients
Uses residual accumulation to reduce information loss

QSGD

Stochastic gradient quantization
Reduces communication cost via low-bit encoding
📚 Datasets
MNIST (toy mode)
CIFAR-10 (full mode)

Supports:

IID distribution
Non-IID distribution using Dirichlet sampling
📈 Visualization

Automatically generates:

Accuracy vs Rounds
Loss vs Rounds
Latency vs Rounds
Communication vs Rounds
Accuracy vs Communication trade-off
Accuracy vs Latency trade-off
Active devices per round
🔬 Research Focus

This project enables analysis of:

Communication vs accuracy trade-offs
Impact of hierarchical aggregation
Efficiency of compression techniques
Fairness in device participation
Latency-aware federated learning
🧾 Citation
@article{uav_hfl_qsgd_2026,
  title={Hierarchical Federated Learning in UAV-Assisted IoT Networks with Compression},
  author={Pragya Rathal},
  journal={IEEE Transactions (Target)},
  year={2026}
}
