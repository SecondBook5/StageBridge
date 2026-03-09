# Target Recommendation Report

This report summarizes whether each edge should be treated as a binary classification task, a continuous risk target, a descriptive-only analysis, or excluded from formal supervised benchmarking.

## AAH->AIS
- Recommended target: `continuous_risk`
- Binary viable: `False`
- Continuous viable: `True`
- Positive donors: `6`
- Negative donors: `1`
- Reason: Continuous risk target is supported by lesion count, donor count, and score diversity.

## AIS->MIA
- Recommended target: `binary_classification`
- Binary viable: `True`
- Continuous viable: `True`
- Positive donors: `8`
- Negative donors: `3`
- Reason: Both classes have enough donor support for donor-held-out binary evaluation.
