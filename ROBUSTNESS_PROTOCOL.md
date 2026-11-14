# MarkRoute robustness replay

These retrospective sensitivity checks reuse five already evaluated floods and the original frozen model construction. They do not form another confirmatory holdout or justify changing the model. Neither sensitivity adds locations or changes the original quality, datum, or deduplication filters. Water-surface elevation remains the outcome.

## Initial survey allowance

Replay 16 original seeds separately for each storm. Keep the 15% partition identical to the original benchmark. Within each event and environment, deterministically shuffle the original anchors and the remaining sites separately. Put the anchor group first. At each nominal allowance of 5%, 10%, 15%, 25%, and 40%, take a prefix using the original minimum of two anchors where possible and retaining one hidden site per environment. This creates nested anchor sets. Both models receive the identical labels at every allowance. They use the original training data and parameters. No outcome-dependent eligibility, weights, or settings change across allowances.

Score two explicit sets. The operational replay scores every location still hidden at that allowance. The common-target replay scores only the sites that remain hidden even at 40%. The 15% operational replay must reproduce both existing holdout reports. Report actual survey counts, MAE in feet, >5-ft rates, 90th and 99th error percentiles, per-event results, and the paired spatial-block bootstrap interval at each allowance. The common-target replay estimates how added readings affect the same sites. A larger allowance is an abstract count of successful readings, with no travel or time equivalence assumed.

## Missing geographic cluster

Use a separate fixed seed stream. For each event and seed, choose one center uniformly among marked locations by outcome-independent random order. Around that center define a compact patch as its closest 20% of the event marks. The first eligible center leaves enough sites outside the patch to fill all original environment-specific 15% quotas. Reserve half of the patch locations, sampled without replacement, as the shared scoring set. Both scenarios exclude scored locations from their initial survey. The dispersed scenario may survey the rest of the patch. The missing-cluster scenario may survey only beyond the patch and fills the same survey quotas. Both methods use the same anchors under each scenario. This evaluates the role of coverage gaps. It does not estimate where travel stops, where crews would survey, or how ground equipment fails.

Evaluate the chosen gradient-boosting mapper and development-selected bounded local regression. Intervals use the original within-event 10-km spatial block bootstrap. Repeat masks reuse locations, and five archived events do not define a population of future floods. Time the frozen refit and prediction/report pipeline with a monotonic clock, reporting execution environment separately from deterministic scores.

In a flat repository, run `python sensitivity.py` beside the existing model code and panel/report fixtures. It writes `robustness_results.json` and `robustness_timing.json`. The nested development directory uses `results.json` and `timing.json` instead. Both layouts are offline and use the same pinned numerical dependencies as the main benchmark.
