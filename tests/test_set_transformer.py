import torch

from stagebridge.models.stagebridge import StageBridgeModel
from stagebridge.utils.types import StageBridgeConfig



def test_set_context_permutation_invariance():
    cfg = StageBridgeConfig(input_dim=16, hidden_dim=32, num_heads=4, num_inducing_points=8)
    model = StageBridgeModel(cfg)
    model.eval()

    x = torch.randn(64, 16)
    perm = torch.randperm(x.shape[0])

    with torch.no_grad():
        c1 = model.forward_set_context(x)
        c2 = model.forward_set_context(x[perm])

    assert c1.shape == c2.shape == (1, cfg.hidden_dim)
    assert torch.allclose(c1, c2, atol=1e-5)
