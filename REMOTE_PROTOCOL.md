# MarkRoute remote-region holdout

Freeze the existing quantized gradient-boosted point model, bounded local regression, original feature extraction and 16 nominal 15% reveal seeds. Fit using the original four training floods alone. Do not use any newly screened event to change the model, outcome, reveal budgets, comparator, or inclusion criteria.

From the same USGS event catalog, screen event type 1 with a recorded start on or after January 1, 2014 and before Helene’s September 24, 2024 start, and nonmissing end by December 31, 2024. Thus these older floods are outside the prior external eligibility rule. Exclude the original eight, every one of the prior 21 screened events, and IDs with archived local raw records before this run. Omit catalog titles containing `test`, `training`, `shakedown`, `bridge`, `gages`, `study`, or `ToT` as independent words. Include every remaining event regardless of region before retrieving its source data.

Apply the original NAVD88, excellent/good/fair, coastal/riverine, distinct site and grid-cell filters. Require at least 30 retained marks. At least 75% must be more than 100 km by great-circle distance from every previous original or first-external study site, using only coordinates. An event with unavailable source terrain is ineligible, with reasons recorded. Archive the eligibility manifest before scoring. Elevation values are never used for event eligibility beyond presence and recorded metadata.

Compare both methods on all retained sites in each qualifying event at 16 shared reveal masks. Report a separate predeclared distance-restricted scoring slice where the nearest prior study site is more than 100 km. Keep within-event observed anchors in both cases. Report MAE, >5-ft misses, tail errors, and spatial-paired intervals. The original training data and any first external outcomes remain unchanged. This tests retrospective estimates of surveyed peak water-surface elevation at known flags, conditional on each event’s available observations. It does not test pre-flood forecasts, inundation, depth, or safe travel.

Report the gradient point model as passing the remote-region stress test only if it reduces pooled MAE by at least 10%, the paired within-event spatial interval has a positive lower bound, and no qualifying event is more than 5% worse. Whether it passes or fails, give all event results and the exclusions.

## Supplementary comparator

After the primary remote benchmark was scored, we compared the archived Florence-and-Michael-selected terrain-aware local kriging configuration on the identical 16 remote masks. Use the selection and model source hashes from `kriging_selection.json` and the original four-storm point model. Report every event, the pooled and event-averaged values, and a paired spatial-block interval. The supplement leaves the primary gate unchanged and does not retune either model.

## Reproduce from packaged data

From the repository root with the pinned requirements installed, run `python replay_remote.py` for the primary results and `python replay_remote.py kriging` for the supplement. These commands use the bundled sanitized `remote_panel.csv`; no network access is needed. Source metadata, record counts, exclusions and terrain hashes are recorded in the remote manifests.
