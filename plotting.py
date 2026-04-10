### plotting.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHOD_COLORS = {
    'A_Standard_FL':        '#1f77b4',
    'B_Clustered_FL':       '#ff7f0e',
    'C_TopK_EF':            '#2ca02c',
    'D_QSGD':               '#d62728',
    'E_TopK_Quorum':        '#9467bd',
    'F_QSGD_Quorum':        '#8c564b',
}

METHOD_LABELS = {
    'A_Standard_FL':        'A: Standard FL',
    'B_Clustered_FL':       'B: Clustered FL',
    'C_TopK_EF':            'C: Cluster+TopK+EF',
    'D_QSGD':               'D: Cluster+QSGD',
    'E_TopK_Quorum':        'E: Cluster+TopK+Quorum',
    'F_QSGD_Quorum':        'F: Cluster+QSGD+Quorum',
}


def _get_color(name):
    return METHOD_COLORS.get(name, '#333333')


def _get_label(name):
    return METHOD_LABELS.get(name, name)


def plot_accuracy(all_histories, mode, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in all_histories.items():
        rounds = [h['round'] for h in history]
        accs = [h['accuracy'] for h in history]
        ax.plot(rounds, accs, label=_get_label(name),
                color=_get_color(name), linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Round', fontsize=13)
    ax.set_ylabel('Test Accuracy', fontsize=13)
    ax.set_title(f'Test Accuracy vs Rounds ({mode.upper()})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    path = os.path.join(save_dir, f'{mode}_accuracy_vs_rounds.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_loss(all_histories, mode, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in all_histories.items():
        rounds = [h['round'] for h in history]
        losses = [h['loss'] for h in history]
        ax.plot(rounds, losses, label=_get_label(name),
                color=_get_color(name), linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Round', fontsize=13)
    ax.set_ylabel('Training Loss', fontsize=13)
    ax.set_title(f'Training Loss vs Rounds ({mode.upper()})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{mode}_loss_vs_rounds.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_latency(all_histories, mode, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in all_histories.items():
        rounds = [h['round'] for h in history]
        lats = [h['latency'] for h in history]
        ax.plot(rounds, lats, label=_get_label(name),
                color=_get_color(name), linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Round', fontsize=13)
    ax.set_ylabel('Latency (s)', fontsize=13)
    ax.set_title(f'Latency vs Rounds ({mode.upper()})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{mode}_latency_vs_rounds.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_communication(all_histories, mode, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in all_histories.items():
        rounds = [h['round'] for h in history]
        # Cumulative comm
        cumcomm = np.cumsum([h['communication_MB'] for h in history]).tolist()
        ax.plot(rounds, cumcomm, label=_get_label(name),
                color=_get_color(name), linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Round', fontsize=13)
    ax.set_ylabel('Cumulative Communication (MB)', fontsize=13)
    ax.set_title(f'Communication vs Rounds ({mode.upper()})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{mode}_communication_vs_rounds.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_tradeoff_accuracy_comm(all_summaries, mode, save_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, metrics in all_summaries.items():
        ax.scatter(
            metrics['total_communication_MB'],
            metrics['best_accuracy'],
            label=_get_label(name),
            color=_get_color(name),
            s=120, zorder=5
        )
        ax.annotate(name.split('_')[0],
                    (metrics['total_communication_MB'], metrics['best_accuracy']),
                    textcoords='offset points', xytext=(5, 5), fontsize=9)
    ax.set_xlabel('Total Communication (MB)', fontsize=13)
    ax.set_ylabel('Best Accuracy', fontsize=13)
    ax.set_title(f'Accuracy-Communication Trade-off ({mode.upper()})', fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{mode}_tradeoff_acc_comm.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_tradeoff_accuracy_latency(all_summaries, mode, save_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, metrics in all_summaries.items():
        ax.scatter(
            metrics['average_latency'],
            metrics['best_accuracy'],
            label=_get_label(name),
            color=_get_color(name),
            s=120, zorder=5
        )
        ax.annotate(name.split('_')[0],
                    (metrics['average_latency'], metrics['best_accuracy']),
                    textcoords='offset points', xytext=(5, 5), fontsize=9)
    ax.set_xlabel('Average Latency (s)', fontsize=13)
    ax.set_ylabel('Best Accuracy', fontsize=13)
    ax.set_title(f'Accuracy-Latency Trade-off ({mode.upper()})', fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{mode}_tradeoff_acc_lat.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_active_devices(all_histories, mode, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in all_histories.items():
        rounds = [h['round'] for h in history]
        active = [h['active_devices'] for h in history]
        ax.plot(rounds, active, label=_get_label(name),
                color=_get_color(name), linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Round', fontsize=13)
    ax.set_ylabel('Active Devices', fontsize=13)
    ax.set_title(f'Active Devices per Round ({mode.upper()})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{mode}_active_devices.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def generate_all_plots(all_histories, all_summaries, mode, results_dir='results'):
    save_dir = os.path.join(results_dir, mode)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nGenerating plots for mode={mode}...")
    plot_accuracy(all_histories, mode, save_dir)
    plot_loss(all_histories, mode, save_dir)
    plot_latency(all_histories, mode, save_dir)
    plot_communication(all_histories, mode, save_dir)
    plot_active_devices(all_histories, mode, save_dir)
    plot_tradeoff_accuracy_comm(all_summaries, mode, save_dir)
    plot_tradeoff_accuracy_latency(all_summaries, mode, save_dir)
    print(f"All plots saved to {save_dir}/")
