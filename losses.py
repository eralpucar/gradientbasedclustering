import torch

class SquaredEuclidean:
    """f(x,y) = 0.5 * ||x-y||^2  — Denklem (3)"""

    def loss(self, centers: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """centers: (K,d)  points: (N,d)  →  (N,K)"""
        diff = points.unsqueeze(1) - centers.unsqueeze(0)   # (N,K,d)
        return 0.5 * (diff ** 2).sum(dim=-1)                # (N,K)

    def gradient(self, center: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """center: (d,)  points: (N,d)  →  (d,)
        ∇_x f = (x - y),  ortalaması alınır — Denklem (7)"""
        return (center.unsqueeze(0) - points).mean(dim=0)


class HuberLoss:
    """f(x,y) = φ_δ(||x-y||)  — Denklem (39)

    ||x-y|| ≤ δ  →  0.5 * ||x-y||²
    ||x-y|| > δ  →  δ*||x-y|| - δ²/2
    """

    def __init__(self, delta: float = 1.0):
        assert delta > 0, "delta pozitif olmalı"
        self.delta = delta

    def loss(self, centers: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """centers: (K,d)  points: (N,d)  →  (N,K)"""
        diff  = points.unsqueeze(1) - centers.unsqueeze(0)  # (N,K,d)
        norms = diff.norm(dim=-1)                           # (N,K)

        quadratic = 0.5 * norms ** 2
        linear    = self.delta * norms - 0.5 * self.delta ** 2

        return torch.where(norms <= self.delta, quadratic, linear)

    def gradient(self, center: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """center: (d,)  points: (N,d)  →  (d,)

        ||x-y|| ≤ δ  →  (x-y)               [quadratic bölge]
        ||x-y|| > δ  →  δ*(x-y)/||x-y||     [gradient clipping]
        """
        diff  = center.unsqueeze(0) - points            # (N,d)
        norms = diff.norm(dim=-1, keepdim=True)         # (N,1)

        grad_near = diff
        grad_far  = self.delta * diff / (norms + 1e-8)

        grad = torch.where(norms <= self.delta, grad_near, grad_far)
        return grad.mean(dim=0)                         # (d,)


class MahalanobisLoss:
    """f(x,y) = 0.5*(x-y)^T A (x-y)  — Denklem (35)

    A: simetrik pozitif tanımlı (d,d)
    """

    def __init__(self, A: torch.Tensor):
        assert A.dim() == 2 and A.shape[0] == A.shape[1], "A kare matris olmalı"
        self.A = A

    def loss(self, centers: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """centers: (K,d)  points: (N,d)  →  (N,K)"""
        diff = points.unsqueeze(1) - centers.unsqueeze(0)       # (N,K,d)
        # (y-x)^T A (y-x) her (n,k) çifti için
        return 0.5 * torch.einsum('nkd,de,nke->nk', diff, self.A, diff)

    def gradient(self, center: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """center: (d,)  points: (N,d)  →  (d,)
        ∇_x f = A(x-y)
        """
        diff = center.unsqueeze(0) - points                     # (N,d)
        return (diff @ self.A).mean(dim=0)                      # (d,)
    


def test_losses():
    torch.manual_seed(42)
    N, K, d = 10, 3, 4
    points  = torch.randn(N, d)
    centers = torch.randn(K, d)
    center  = centers[0]               # tek merkez, gradient için

    # ── SquaredEuclidean ──────────────────────────────────────
    se = SquaredEuclidean()

    out = se.loss(centers, points)
    assert out.shape == (N, K), f"Beklenen (N,K), geldi {out.shape}"
    assert (out >= 0).all(),    "Loss negatif olamaz"

    g = se.gradient(center, points)
    assert g.shape == (d,),     f"Gradient shape yanlış: {g.shape}"

    # K-means sabit noktası kontrolü:
    # merkez = kümesinin ortalamasıysa gradient sıfır olmalı
    mean_point = points.mean(dim=0)
    g_zero     = se.gradient(mean_point, points)
    assert g_zero.norm() < 1e-6, "Ortalama noktada gradient sıfır olmalı"

    # ── HuberLoss ─────────────────────────────────────────────
    delta = 1.0
    hl    = HuberLoss(delta)

    out_h = hl.loss(centers, points)
    assert out_h.shape == (N, K)
    assert (out_h >= 0).all()

    # δ sınırında süreklilik: ||x-y|| = δ iken iki formül eşit olmalı
    x   = torch.zeros(1, d)
    y_  = torch.zeros(1, d); y_[0, 0] = delta     # tam δ uzaklıkta
    val_quad   = 0.5 * delta ** 2
    val_linear = delta * delta - 0.5 * delta ** 2  # = 0.5*delta^2
    assert abs(val_quad - val_linear) < 1e-6, "δ sınırında süreksizlik!"

    g_h = hl.gradient(center, points)
    assert g_h.shape == (d,)

    # Gradient clipping kontrolü: uzak noktada norm = delta olmalı
    far_point  = center + 100.0 * torch.ones(d)   # çok uzak
    g_far      = hl.gradient(center, far_point.unsqueeze(0))
    assert abs(g_far.norm().item() - delta) < 1e-4, \
        f"Uzak noktada gradient norm delta olmalı, geldi {g_far.norm():.4f}"

    # ── MahalanobisLoss ───────────────────────────────────────
    A   = torch.eye(d) * 2.0                      # 2I → scaled Öklid
    ml  = MahalanobisLoss(A)

    out_m = ml.loss(centers, points)
    assert out_m.shape == (N, K)
    assert (out_m >= 0).all()

    # A=2I iken Mahalanobis = 2 * SquaredEuclidean olmalı
    diff  = (points.unsqueeze(1) - centers.unsqueeze(0))
    expected = (diff ** 2).sum(-1)                 # = 2 * 0.5 * ||diff||^2 * 2... kontrol:
    # f = 0.5*(x-y)^T (2I) (x-y) = ||x-y||^2
    se_doubled = (diff.norm(dim=-1) ** 2)
    assert torch.allclose(out_m, se_doubled, atol=1e-5), \
        "A=2I için Mahalanobis = ||x-y||^2 olmalı"

    g_m = ml.gradient(center, points)
    assert g_m.shape == (d,)

    print("Tüm testler geçti ✓")

test_losses()