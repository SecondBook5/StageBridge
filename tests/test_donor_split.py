from stagebridge.training.trainer import build_donor_holdout_splits



def test_donor_holdout_split_no_overlap():
    donors = [f"D{i}" for i in range(12)]
    splits = build_donor_holdout_splits(donor_ids=donors, n_folds=3, seed=42)

    for split in splits:
        train = set(split.train_donors)
        val = set(split.val_donors)
        test = set(split.test_donors)

        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
