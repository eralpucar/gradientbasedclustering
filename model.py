import torch
import time
from itertools import permutations
from typing import Optional
from scipy.optimize import linear_sum_assignment
from .assignment import assign_clusters, euclidean_distance


class GradientClustering:
    """Gradient Based Clustering — Algorithm (6)-(7), Armacki et al. ICML 2022"""

    def __init__(self,
                 K:            int,
                 loss_fn,
                 distance_fn   = None,
                 alpha:         float = None,
                 max_iter:      int   = 300,
                 tol:           float = 1e-6,
                 init:          str   = 'kmeans++'):
        self.K            = K
        self.loss_fn      = loss_fn
        self.distance_fn  = distance_fn or euclidean_distance
        self.alpha        = alpha
        self.max_iter     = max_iter
        self.tol          = tol
        self.init         = init

        self.centers_          = None
        self.labels_           = None
        self.loss_history_     = []
        self.accuracy_history_ = []

    # ── Init ──────────────────────────────────────────────────────────────

    def _init_centers(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        N, d = X.shape

        if self.init == 'random':
            return X[torch.randperm(N)[:self.K]].clone()

        elif self.init == 'kmeans++':
            centers = [X[torch.randint(N, (1,)).item()].clone()]
            for _ in range(self.K - 1):
                # her noktanın en yakın mevcut merkeze uzaklığı
                dists = torch.stack(
                    [(X - c).norm(dim=1) for c in centers], dim=1
                ).min(dim=1).values                     # (N,)
                probs = dists ** 2
                probs /= probs.sum()
                centers.append(X[torch.multinomial(probs, 1).item()].clone())
            return torch.stack(centers)                 # (K, d)

        elif self.init == 'from_class':
            # Paper'ın tam setup'ı: her sınıftan 1 rastgele nokta — Section VI
            if y is None:
                raise ValueError("from_class init için y_true gerekli")
            centers = []
            for k in range(self.K):
                idx = (y == k).nonzero(as_tuple=True)[0]
                if len(idx) == 0:
                    raise ValueError(f"Sınıf {k} veri setinde yok")
                pick = idx[torch.randint(len(idx), (1,)).item()]
                centers.append(X[pick].clone())
            return torch.stack(centers)                 # (K, d)

        else:
            raise ValueError(f"Bilinmeyen init: {self.init}")

    # ── Ana döngü ─────────────────────────────────────────────────────────

    def fit(self, X: torch.Tensor, y_true: Optional[torch.Tensor] = None):
        """
        X:      (N, d)
        y_true: (N,)  opsiyonel — accuracy takibi ve from_class init için
        """
        # Her fit() başında geçmişi sıfırla
        self.loss_history_     = []
        self.accuracy_history_ = []

        N     = X.shape[0]
        alpha = self.alpha if self.alpha is not None else 1.0 / N

        centers = self._init_centers(X, y_true)
        self.init_centers_ = centers.clone()  

        for t in range(self.max_iter):

            # Adım 1 — Küme atama: Denklem (6)
            assignments = assign_clusters(X, centers, self.distance_fn)

            # Loss kaydı (Lemma 1: monoton azalıyor mu kontrol)
            loss = self._compute_loss(X, centers, assignments)
            self.loss_history_.append(loss.item())

            if y_true is not None:
                self.accuracy_history_.append(
                    self._cluster_accuracy(assignments, y_true)
                )

            # Adım 2 — Merkez güncelleme: Denklem (7)
            new_centers = centers.clone()
            for i in range(self.K):
                mask = (assignments == i)
                if mask.sum() == 0:
                    continue                    # boş küme — merkezi koru
                grad = self.loss_fn.gradient(centers[i], X[mask])
                new_centers[i] = centers[i] - alpha * grad

            # Yakınsama: Theorem 1 — merkez kayması tol altına düşünce dur
            shift = (new_centers - centers).norm().item()
            centers = new_centers

            if shift < self.tol:
                print(f"  Yakınsadı — iterasyon {t + 1}, shift={shift:.2e}")
                break

        self.centers_ = centers
        self.labels_  = assign_clusters(X, centers, self.distance_fn)
        return self

    # ── Yardımcı ──────────────────────────────────────────────────────────

    def _compute_loss(self, X, centers, assignments):
        """J(x, C) = Σ_i Σ_{y∈C(i)} p_y f(x(i), y) — Denklem (4)
        Uniform ağırlık: p_y = 1/N
        """
        N     = X.shape[0]
        total = torch.tensor(0.0)
        for i in range(self.K):
            mask = (assignments == i)
            if mask.sum() == 0:
                continue
            # loss() (K,d) ve (N,d) bekliyor — tek merkezi (1,d) olarak gönder
            vals = self.loss_fn.loss(
                centers[i].unsqueeze(0),   # (1, d)
                X[mask]                    # (n_i, d)
            )                              # → (n_i, 1)
            total += vals.sum() / N
        return total

    def _cluster_accuracy(self,
                          pred: torch.Tensor,
                          true: torch.Tensor) -> float:
        """En iyi etiket eşleşmesini Hungarian algoritmasıyla bul.

        K=7 için 7!=5040 permütasyon yerine O(K³) Hungarian kullanır.
        Paper'ın "tüm permütasyonları dene" yöntemiyle matematiksel olarak
        eşdeğer sonuç verir, çok daha hızlı.
        """
        K = self.K
        pred_np = pred.numpy()
        true_np = true.numpy()

        # Maliyet matrisi: cost[i,j] = sınıf i'yi küme j'ye atarsak yanlış sayısı
        cost = torch.zeros(K, K, dtype=torch.long)
        for i in range(K):
            for j in range(K):
                cost[i, j] = ((pred_np == j) & (true_np == i)).sum()

        # Maksimum eşleşme = minimum negatif maliyet
        row_ind, col_ind = linear_sum_assignment(-cost.numpy())
        correct = cost[row_ind, col_ind].sum().item()
        return correct / len(true_np)
    
def test_clustering():
    from sklearn.datasets import load_iris
    from losses import SquaredEuclidean

    data   = load_iris()
    X      = torch.tensor(data.data).float()
    y      = torch.tensor(data.target)

    gc = GradientClustering(K=3, loss_fn=SquaredEuclidean(),
                            init='from_class', max_iter=200)
    gc.fit(X, y_true=y)

    # Lemma 1 kontrolü: loss monoton azalmalı
    losses = gc.loss_history_
    for i in range(1, len(losses)):
        assert losses[i] <= losses[i-1] + 1e-6, \
            f"Lemma 1 ihlali: iterasyon {i}, {losses[i]:.6f} > {losses[i-1]:.6f}"

    print(f"Final accuracy : {gc.accuracy_history_[-1]:.3f}")
    print(f"İterasyon sayısı: {len(losses)}")
    print(f"Loss monoton azaldı: ✓")

#test_clustering()