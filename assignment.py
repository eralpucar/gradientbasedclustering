def assign_clusters(points, centers, distance_fn):
    """Voronoi atama — Denklem (6)
    
    Her y ∈ D için: argmin_i  g(x(i), y)
    
    points:  (N, d)
    centers: (K, d)
    döner:   (N,) — her noktanın küme indeksi
    """
    # distance_fn(centers, points) → (N, K) mesafe matrisi
    distances   = distance_fn(centers, points)  # (N, K)
    assignments = distances.argmin(dim=1)        # (N,)
    return assignments


def euclidean_distance(centers, points):
    """g(x,y) = ||x - y||"""
    diff = points.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, d)
    return diff.norm(dim=-1)                            # (N, K)


def mahalanobis_distance(A):
    """g(x,y) = ||x-y||_A = sqrt((x-y)^T A (x-y))"""
    def _dist(centers, points):
        diff = points.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, d)
        return (0.5 * (diff @ A * diff).sum(dim=-1)).sqrt()
    return _dist