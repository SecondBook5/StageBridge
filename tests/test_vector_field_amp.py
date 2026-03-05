import pytest
import torch

from stagebridge.models.stagebridge import StageBridgeModel
from stagebridge.utils.types import StageBridgeConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP stability test requires CUDA")
def test_vector_field_amp_shape_and_dtype_stable():
    cfg = StageBridgeConfig(input_dim=16, hidden_dim=32, num_heads=4, device="cuda")
    model = StageBridgeModel(cfg).cuda().eval()

    x_set = torch.randn(1, 64, 16, device="cuda")
    x_t = torch.randn(128, 16, device="cuda")
    t = torch.rand(128, device="cuda")

    with torch.no_grad():
        c = model.forward_set_context(x_set)
        sid = model.encode_stage_pair_tensor(0, 1, n=x_t.shape[0], device=x_t.device)
        with torch.amp.autocast("cuda", enabled=True):
            v = model.forward_vector_field(x_t=x_t, t=t, c_s=c, stage_pair_id=sid)

    assert v.shape == x_t.shape
    assert torch.isfinite(v).all()
