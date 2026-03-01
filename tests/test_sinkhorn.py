import torch

from stagebridge.training.losses import build_sinkhorn_coupling



def test_sinkhorn_valid_marginals_and_finite():
    torch.manual_seed(0)
    x = torch.randn(32, 8)
    y = torch.randn(40, 8)
    pi = build_sinkhorn_coupling(x, y, epsilon=0.1, n_iters=100)

    assert torch.isfinite(pi).all()
    assert abs(float(pi.sum().item()) - 1.0) < 1e-4

    row = pi.sum(dim=1)
    col = pi.sum(dim=0)
    assert torch.allclose(row, torch.full_like(row, 1.0 / x.shape[0]), atol=5e-3)
    assert torch.allclose(col, torch.full_like(col, 1.0 / y.shape[0]), atol=5e-3)
