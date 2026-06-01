# reproduce.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torchvision import datasets, transforms

from src.losses import SquaredEuclidean, HuberLoss
from src.model import GradientClustering
from src.assignment import assign_clusters, euclidean_distance


# ─────────────────────────────────────────────────────────────
# Veri yükleme
# ─────────────────────────────────────────────────────────────

def load_mnist_subset():
    """Paper Section VI: rakamlar 1-7, her sınıftan 500, normalize"""
    ds = datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.ToTensor())
    X_list, y_list = [], []
    counts = {i: 0 for i in range(1, 8)}

    for img, label in ds:
        if label in counts and counts[label] < 500:
            X_list.append(img.flatten())
            y_list.append(label - 1)
            counts[label] += 1
        if all(v == 500 for v in counts.values()):
            break

    X = torch.stack(X_list).float() / 255.0
    y = torch.tensor(y_list)
    print(f"MNIST yüklendi: {X.shape}")
    return X, y


def load_iris():
    from sklearn.datasets import load_iris as _load
    data = _load()
    X = torch.tensor(data.data).float()
    y = torch.tensor(data.target)
    print(f"Iris yüklendi: {X.shape}")
    return X, y


def add_noise(X, noise_pct: float, noise_var: float, seed: int = None):
    if seed is not None:
        torch.manual_seed(seed)
    N = len(X)
    n_noisy = int(N * noise_pct)
    idx = torch.randperm(N)[:n_noisy]
    X_noisy = X.clone()
    
    # Gürültüyü veri ölçeğine normalize et
    data_std = X.std().item()
    noise_std = (noise_var ** 0.5) * data_std   # ← veri std'siyle ölçekle
    
    X_noisy[idx] += torch.randn(n_noisy, X.shape[1]) * noise_std
    return X_noisy


# ─────────────────────────────────────────────────────────────
# Yardımcı: Hungarian accuracy
# ─────────────────────────────────────────────────────────────

def hungarian_accuracy(pred: torch.Tensor, true: torch.Tensor, K: int) -> float:
    cost = torch.zeros(K, K, dtype=torch.long)
    for i in range(K):
        for j in range(K):
            cost[i, j] = ((pred.numpy() == j) & (true.numpy() == i)).sum()
    row, col = linear_sum_assignment(-cost.numpy())
    return cost[row, col].sum().item() / len(true)


# ─────────────────────────────────────────────────────────────
# Yardımcı: sklearn KMeans iterasyon bazlı accuracy curve
# ─────────────────────────────────────────────────────────────

def kmeans_accuracy_curve(X: torch.Tensor, y: torch.Tensor,
                          init_centers: torch.Tensor,
                          K: int, max_iter: int) -> list:
    centers = init_centers.numpy().copy()
    acc_curve = []

    for it in range(1, max_iter + 1):
        km = KMeans(n_clusters=K, init=centers, n_init=1, max_iter=1)
        km.fit(X.numpy())
        centers = km.cluster_centers_
        pred = torch.tensor(km.labels_)
        acc_curve.append(hungarian_accuracy(pred, y, K))

        if it > 5 and abs(acc_curve[-1] - acc_curve[-2]) < 1e-7:
            acc_curve += [acc_curve[-1]] * (max_iter - it)
            break

    return acc_curve


# ─────────────────────────────────────────────────────────────
# Yardımcı: Pediredla [2] fixed-point güncellemesi
# ─────────────────────────────────────────────────────────────

def pediredla_update(centers: torch.Tensor, X: torch.Tensor,
                     assignments: torch.Tensor, K: int, delta: float):
    new_centers = centers.clone()
    for i in range(K):
        mask = (assignments == i)
        if mask.sum() == 0:
            continue
        pts  = X[mask]
        diff = (centers[i].unsqueeze(0) - pts).norm(dim=1)

        near = diff <= delta
        far  = ~near

        num = torch.zeros_like(centers[i])
        den = torch.tensor(0.0)

        if near.sum() > 0:
            num += pts[near].sum(dim=0)
            den += near.sum().float()

        if far.sum() > 0:
            w    = delta / (diff[far] + 1e-8)
            num += (w.unsqueeze(1) * pts[far]).sum(dim=0)
            den += w.sum()

        if den > 0:
            new_centers[i] = num / den

    return new_centers


def run_pediredla(X: torch.Tensor, y: torch.Tensor,
                  init_centers: torch.Tensor,
                  K: int, delta: float, max_iter: int) -> list:
    centers   = init_centers.clone()
    acc_curve = []

    for _ in range(max_iter):
        assignments = assign_clusters(X, centers, euclidean_distance)
        centers     = pediredla_update(centers, X, assignments, K, delta)
        acc_curve.append(hungarian_accuracy(assignments, y, K))

    return acc_curve


# ─────────────────────────────────────────────────────────────
# Plot yardımcıları
# ─────────────────────────────────────────────────────────────

def plot_on_ax(ax, curves_a, label_a, color_a,
               curves_b, label_b, color_b, title):

    iters = np.arange(1, curves_a.shape[1] + 1)

    global_min = float('inf')
    global_max = float('-inf')

    for curves, label, color, marker in [
        (curves_a, label_a, color_a, 'o'),
        (curves_b, label_b, color_b, '>'),
    ]:
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)

        lower = mean - std
        upper = mean + std

        global_min = min(global_min, lower.min())
        global_max = max(global_max, upper.max())

        ax.plot(
            iters,
            mean,
            color=color,
            label=label,
            marker=marker,
            markevery=max(1, len(iters) // 8),
            markersize=5,
            linewidth=1.5,
        )

        ax.fill_between(
            iters,
            lower,
            upper,
            color=color,
            alpha=0.15,
        )

    # margin ekle
    y_min = max(0.0, global_min - 0.05)
    y_max = min(1.0, global_max + 0.05)

    ax.set_ylim(y_min, y_max)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('İterasyon')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)


def pad_curves(curves: list, length: int) -> np.ndarray:
    return np.array([c + [c[-1]] * (length - len(c)) for c in curves])


# ─────────────────────────────────────────────────────────────
# Deney 1 — Fig 1 & 2: Gradient K-means vs K-means
# ─────────────────────────────────────────────────────────────

def experiment_fig1_fig2(dataset: str = 'mnist',
                          n_runs:  int  = 20,
                          max_iter: int = 100):
    """Paper Fig 1 (MNIST) ve Fig 2 (Iris)"""

    if dataset == 'mnist':
        X, y   = load_mnist_subset()
        K      = 7
        fig_no = '1'
        alpha = 0.1
    else:
        X, y   = load_iris()
        K      = 3
        fig_no = '2'
        alpha  = 0.1

    N = len(X)

    all_grad, all_kmeans = [], []

    for run in range(n_runs):
        print(f"  [{dataset.upper()}] Fig {fig_no} — run {run+1}/{n_runs}", end='\r')
        torch.manual_seed(run)

        gc = GradientClustering(
            K=K,
            loss_fn=SquaredEuclidean(),
            alpha=alpha,
            max_iter=max_iter,
            init='from_class' if dataset == 'mnist' else 'random',
        )
        gc.fit(X, y_true=y)
        all_grad.append(gc.accuracy_history_)

        km_curve = kmeans_accuracy_curve(
            X, y,
            init_centers=gc.init_centers_,
            K=K,
            max_iter=max_iter,
        )
        all_kmeans.append(km_curve)

    print()

    all_grad   = pad_curves(all_grad,   max_iter)
    all_kmeans = pad_curves(all_kmeans, max_iter)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_on_ax(
        ax,
        curves_a=all_grad,   label_a='Gradient K-means', color_a='#378ADD',
        curves_b=all_kmeans, label_b='K-means',           color_b='#E24B4A',
        title=f'Fig {fig_no}: {dataset.upper()} — Gradient K-means vs K-means',
    )
    plt.tight_layout()
    path = f'outputs/fig{fig_no}_{dataset}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Kaydedildi: {path}")
    plt.show()

    return all_grad, all_kmeans


def experiment_fig3_fig4(dataset:  str = 'mnist',
                          n_runs:   int = 20,
                          max_iter: int = 140):
    """Paper Fig 3 (MNIST) ve Fig 4 (Iris)
    2×2 grid: noise_pct ∈ {0.10, 0.20} × noise_var ∈ {1.0, 2.0}
    """

    if dataset == 'mnist':
        X_clean, y = load_mnist_subset()
        K          = 7
        delta      = 5      # normalize [0,1] veri için — 10.0 çok büyüktü
        alpha      = 0.1
        fig_no     = '3'
    else:
        X_clean, y = load_iris()
        K          = 3
        delta      = 5.0
        alpha      = 0.1
        fig_no     = '4'

    N = len(X_clean)
    noise_pcts = [0.10, 0.20]
    noise_vars = [1.0,  2.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    fig.suptitle(
        f'Fig {fig_no}: {dataset.upper()} — Huber Gradient vs Pediredla [2]',
        fontsize=13,
    )

    for row, noise_pct in enumerate(noise_pcts):
        for col, noise_var in enumerate(noise_vars):

            all_huber, all_ped = [], []

            for run in range(n_runs):
                print(f"  [{dataset.upper()}] Fig {fig_no} — "
                      f"pct={int(noise_pct*100)}% var={noise_var} "
                      f"run {run+1}/{n_runs}", end='\r')
                torch.manual_seed(run)

                X = add_noise(X_clean, noise_pct, noise_var, seed=run)

                # Shared init: 1 round Lloyd — paper'ın yöntemi
                km_init = KMeans(n_clusters=K, n_init=1, max_iter=1,
                                 random_state=run)
                km_init.fit(X.numpy())
                init_centers = torch.tensor(km_init.cluster_centers_).float()

                # Gradient Huber — Lloyd init ile
                gc = GradientClustering(
                    K=K,
                    loss_fn=HuberLoss(delta=delta),
                    tol=0,
                    alpha=alpha,
                    max_iter=max_iter,
                    init='from_class' if dataset == 'mnist' else 'random', 
                )
                #gc._forced_init  = init_centers.clone()
                #gc._init_centers = lambda Xd, yd=None: gc._forced_init
                gc.fit(X, y_true=y)
                all_huber.append(gc.accuracy_history_)

                # Pediredla [2] — aynı init
                ped_curve = run_pediredla(X, y, init_centers, K, delta, max_iter)
                all_ped.append(ped_curve)
            print()


            plot_on_ax(
                axes[row][col],
                curves_a=pad_curves(all_huber, max_iter),
                label_a='Gradient Huber',
                color_a='#378ADD',
                curves_b=pad_curves(all_ped, max_iter),
                label_b='Huber [2]',
                color_b='#E24B4A',
                title=f'Noise={int(noise_pct*100)}%  Var={noise_var}',
            )

    plt.tight_layout()
    path = f'outputs/fig{fig_no}_{dataset}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Kaydedildi: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Deney 1: Gradient K-means vs K-means ===")
    experiment_fig1_fig2('iris',  n_runs=20, max_iter=300)
    experiment_fig1_fig2('mnist', n_runs=20, max_iter=300)

    print("\n=== Deney 2: Huber Gradient vs Pediredla [2] ===")
    experiment_fig3_fig4('iris',  n_runs=20, max_iter=140)
    experiment_fig3_fig4('mnist', n_runs=20, max_iter=140)