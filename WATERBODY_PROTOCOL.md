# Waterbody ablation

**Hypothesis.** A named-waterbody match helps a learned flood-elevation correction beyond location and terrain alone. The expected benefit is largest for riverine marks, where different channels may be close in straight-line distance. The reported match is a proxy for hydrologic connectivity, which would require a channel-network dataset to establish.

Compare two fits of the same gradient-boosted estimator on the same four training events and 18 masks. For the ablation, blank the waterbody tags throughout training and evaluation, making the three match statistics always zero. Keep terrain, straight-line distances, environment, revealed elevations, the physical starting estimate, all other features, model parameters and masks fixed.

Primary comparison is absolute error on the riverine portion of the 13 evaluation floods with the same starting measurements. Secondary comparisons are coastal and pooled errors. The point-prediction outcomes for the full model are already known. Use a paired bootstrap of floods and 10 km blocks, preserving mask repetitions within blocks. Call a riverine benefit supported if pooled MAE falls by at least 2% and the paired 95% interval is strictly positive. Report subgroup and event counts and all directions regardless of the result.
