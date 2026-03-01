import torch

from stagebridge.models.stagebridge import StageBridgeModel
from stagebridge.training.losses import flow_matching_loss
from stagebridge.utils.types import StageBatch, StageBridgeConfig



def test_flow_matching_loss_decreases_on_toy_problem():
    torch.manual_seed(7)

    cfg = StageBridgeConfig(
        input_dim=8,
        hidden_dim=32,
        vector_field_hidden_dim=64,
        num_heads=4,
        num_inducing_points=8,
        max_epochs=5,
    )
    model = StageBridgeModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    x_src = torch.randn(128, 8)
    x_tgt = x_src + 0.5  # simple translation
    batch = StageBatch(x_src=x_src, x_tgt=x_tgt, stage_src=0, stage_tgt=1, donor_id="D1")

    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss, _, _ = flow_matching_loss(
            batch=batch,
            model=model,
            ot_epsilon=0.1,
            sinkhorn_iters=50,
            num_ot_pairs=128,
            context_consistency_weight=0.0,
            use_ot=True,
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))

    assert losses[-1] < losses[0]
