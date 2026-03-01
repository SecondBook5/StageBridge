import numpy as np
import torch

from stagebridge.training.trainer import build_donor_holdout_splits
from stagebridge.utils.seeds import set_global_seed



def test_deterministic_split_and_torch_seed():
    donors = [f"D{i}" for i in range(9)]
    s1 = build_donor_holdout_splits(donor_ids=donors, n_folds=3, seed=123)
    s2 = build_donor_holdout_splits(donor_ids=donors, n_folds=3, seed=123)
    assert s1 == s2

    set_global_seed(17)
    torch.manual_seed(17)
    a = torch.randn(4, 4)

    set_global_seed(17)
    torch.manual_seed(17)
    b = torch.randn(4, 4)

    assert np.allclose(a.numpy(), b.numpy())
