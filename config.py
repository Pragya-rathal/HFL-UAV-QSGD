### config.py
import argparse
import random
import numpy as np
import torch

def get_config(mode='toy'):
    if mode == 'toy':
        config = {
            'mode': 'toy',
            'dataset': 'mnist',
            'num_devices': 25,
            'num_clusters': 5,
            'num_rounds': 22,
            'local_epochs': 2,
            'batch_size': 32,
            'learning_rate': 0.01,
            'data_distribution': 'iid',   # 'iid' or 'non_iid'
            'dirichlet_alpha': 0.5,
            'seed': 42,
            'quorum_fraction': 0.7,
            'quorum_K': 5,
            'topk_ratio': 0.1,
            'qsgd_levels': 8,
            'image_size': 28,
            'num_classes': 10,
            'in_channels': 1,
        }
    else:
        config = {
            'mode': 'full',
            'dataset': 'cifar10',
            'num_devices': 75,
            'num_clusters': 10,
            'num_rounds': 75,
            'local_epochs': 4,
            'batch_size': 64,
            'learning_rate': 0.01,
            'data_distribution': 'non_iid',
            'dirichlet_alpha': 0.5,
            'seed': 42,
            'quorum_fraction': 0.7,
            'quorum_K': 10,
            'topk_ratio': 0.1,
            'qsgd_levels': 8,
            'image_size': 32,
            'num_classes': 10,
            'in_channels': 3,
        }
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Hierarchical Federated Learning for UAV-IoT')
    parser.add_argument('--mode', type=str, default='toy', choices=['toy', 'full'],
                        help='Run mode: toy (MNIST) or full (CIFAR-10)')
    return parser.parse_args()
