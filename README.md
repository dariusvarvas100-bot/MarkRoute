# MarkRoute

**AI + Climate · Veritas AI International High School Hackathon 2026**

After a flood, a survey coordinator can have a list of places where crews found high-water marks, with elevations for only some of them. MarkRoute estimates the peak water-surface elevation at the remaining flags. Open the replay at an Ian survey mark and reveal the crew’s recorded 12.30 ft. MarkRoute’s estimate missed by 0.21 ft and bounded regression missed by 1.85 ft. Add the reading to watch the same models recalculate the other 249 marks.

**With the same initial surveys, MarkRoute reduced mean error in 13 of 13 archived floods against bounded terrain regression.** It cut error from **2.48 to 2.08 ft** on Ian and Helene and from **6.17 to 5.30 ft** on three additional floods. In a separate geographic test covering eight floods, error fell from **5.46 to 4.57 ft** across 1,371 sites more than 100 km from earlier study sites. It still struggled in Helene, where 49% of the estimates missed by more than five feet.

[Open the live replay](https://dariusvarvas100-bot.github.io/MarkRoute/demo.html) · [Download the offline replay](https://github.com/dariusvarvas100-bot/MarkRoute/archive/refs/heads/main.zip) · [One-page write-up](MarkRoute_Writeup.pdf) · [106-second walkthrough](MarkRoute_Demo.mp4)

The opening replay shows the recalculation at the closest remaining mark in the same environment. Average error across the same 249 remaining marks falls from 1.21 to 1.20 ft. At the closest mark, error rises from 0.69 to 0.72 ft. In the map, choose **Measured miss** to inspect a separate site where bounded regression is closer. Each point shows the starting estimate and the model’s signed correction.

**What we added.** Click **Explain this estimate** to see how the nearest readings changed a prediction, in feet. On Ian mark 32828, 40 fixed surveys give an 11.52 ft estimate. The four nearby readings move it to 12.51 ft. The strongest contribution is **+0.59 ft** from 6.14 km away, even though the nearest reading is 1.18 km away. The crew’s measurement at the target remains hidden during the explanation.

## Held-out results

Same initial measurements for both methods. Lower mean absolute error is better.

| Storm | Bounded local terrain regression (ft) | MarkRoute (ft) | Reduction |
| --- | ---: | ---: | ---: |
| Hurricane Ian, 2022 | 1.32 | 1.15 | 13.3% |
| Hurricane Helene, 2024 | 8.14 | 6.65 | 18.2% |
| Pooled | 2.48 | 2.08 | **16.0%** |

The paired 95% spatial-block interval for the pooled absolute improvement was **0.17 to 0.72 ft**, conditional on these two archived events. The share of errors greater than five feet fell from **12.1% to 10.1%** pooled. Helene remained challenging, with 49% of the errors larger than five feet. The 355 locations contribute 4,816 unobserved location and mask pairs across the two storms. Repeated predictions at one location do not represent independent observations. [Exact results](holdout.json)

## Additional test on three floods

After the initial comparison, we kept the model and baseline fixed and recorded the selection rule for a new set of archived events. Three events had enough eligible marks. Each showed a lower mean error at the same measurement budget.

| Event | Bounded regression (ft) | MarkRoute (ft) | Reduction |
| --- | ---: | ---: | ---: |
| Hurricane Ida, 2021 | 5.58 | 4.79 | 14.1% |
| July 2025 Texas flood | 7.05 | 6.07 | 13.9% |
| December 2025 Pacific Northwest flood | 4.87 | 4.08 | 16.1% |
| Pooled | 6.17 | 5.30 | **14.1%** |

The external check covers **659 archived sites** and 8,960 withheld location and mask pairs. Its paired 95% spatial interval was **0.61 to 1.15 ft** improvement within these events. The catalog screening included a 120-day archive lag and a minimum count of qualifying marks. The [selection protocol](EXTERNAL_PROTOCOL.md) and [full results](external_results.json) list the supported events and exclusions.

## Eight floods far from the earlier study sites

This separate test kept the point model and bounded regression fixed. We screened USGS events using archive completeness, eligible survey counts, and geographic distance before scoring. The eight qualifying floods contributed **1,371 actual sites**, each more than 100 km from the nearest site in the earlier panels. As in the original evaluation, each method started with the same roughly 15% survey sample in each replay.

Mean error fell from **5.46 to 4.57 ft (16.3%)**. The mean improved in all eight floods. The paired spatial interval for the improvement was **0.68 to 1.13 ft** within these archives. Errors larger than five feet fell from **34.8% to 29.3%** across 18,608 withheld predictions. Storms contribute differing numbers of observations, and the same site may occur in several survey splits. [Geographic selection protocol](REMOTE_PROTOCOL.md) · [Event-level results](remote_results.json) · [Data manifest](remote_manifest.json)

## Comparison with spatial kriging

Kriging predicts a location’s elevation using correlations between the surveyed points. The comparison fits a terrain trend, then solves a pooled covariance system for nearby survey residuals. Its correlation decays with geographic separation and incorporates differences in terrain elevation and waterbody. We chose the setting on Florence and Michael and evaluated it on the same initial surveys and withheld points used for the original five-storm test.

| Evaluation | Terrain-aware kriging (ft) | MarkRoute (ft) |
| --- | ---: | ---: |
| Ian and Helene | 2.67 | 2.08 |
| Three additional floods | 6.08 | 5.30 |
| Combined | 4.89 | **4.17** |

MarkRoute’s mean error was lower on all five events and **14.6% lower across the combined observations**. The same frozen kriging configuration also had higher mean error on each of the eight geographic holdout floods. On those sites, MarkRoute cut mean error from **5.46 to 4.57 ft** compared with kriging. [Geographic kriging results](remote_kriging_results.json) The paired spatial interval for the combined improvement was **0.49 to 0.99 ft** within these floods. Kriging was stronger than bounded regression on the three-event cohort. This additional comparator uses the existing archive and was added after the original benchmark. [Protocol](kriging_protocol.md) · [Selection](kriging_selection.json) · [Results](kriging_results.json)

## How the result changes with fewer surveys

At each of five survey budgets, both methods received the same starting readings. We evaluated all budgets on the same 9,744 withheld predictions, so an apparent improvement could not come from measuring the hardest locations and removing them from the test.

| Initial survey budget | Bounded regression (ft) | MarkRoute (ft) |
| --- | ---: | ---: |
| About 5% | 7.76 | 5.28 |
| About 10% | 5.29 | 4.46 |
| About 15% | 4.76 | 4.11 |
| About 25% | 4.24 | 3.72 |
| About 40% | 3.66 | 3.35 |

When the same initial readings were redistributed to leave an entire neighborhood unmeasured, MarkRoute’s average error rose from **4.18 to 5.28 ft**. It still had lower pooled error than bounded regression, which reached **6.63 ft**, while bounded regression had a slight advantage in Ian. These are additional stress tests of the frozen model on the same events. [Protocol](ROBUSTNESS_PROTOCOL.md) · [Full results](robustness_results.json)

The replay can also highlight the approximately 20% of withheld marks where the two estimates disagree most. That rule flagged **1,148 of 3,798** errors larger than five feet across the five floods. Model disagreement was the strongest simple review rule on validation. A separate trained ranker and a rule based on rechecking neighboring measurements caught fewer large errors on the held-out floods. Many large errors remained outside the review queue. [Review replay and results](review_results.json) · [Evaluation script](review.py)

## Which readings moved the estimate?

The offline browser recomputes the live estimator for each of up to 16 subsets of the four closest surveyed readings in the same environment. All other surveyed readings are held fixed, including at least one from that environment. A reading's [Shapley value](https://doi.org/10.1515/9781400881970-018) is its average marginal change across all possible orders of adding those readings. For a subset of size `s` and `m` varied readings, the weight is `s! (m - s - 1)! / m!`. The signed values sum to the difference between the fixed-survey estimate and the full estimate. These values describe how the deployed model responds, given the chosen four nearby readings. They do not represent physical causation or an error bound.

The browser test confirms that the attributions sum correctly in 32 archived replays, are unchanged if hidden answers are erased, and agree with an independent enumeration of all 24 orders in a four-reading example. [Related research on model attribution](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)

## Does geographic structure transfer?

We tested two hypotheses with otherwise identical booster fits and the same revealed marks. First, terrain should improve transfer to riverine sites located more than 100 km from the earlier study sites. Removing terrain from both the starting estimate and learned features raised MAE from **4.58 to 85.00 ft** across **1,365 riverine sites** in eight geographic holdout floods. The complete model had lower mean error in every flood. This is a data-removal experiment. Practical comparators retain terrain and score about **5.46 ft** in the larger geographic benchmark above. [Terrain experiment protocol](TERRAIN_ABLATION_PROTOCOL.md) · [Terrain results](terrain_ablation_results.json)

Second, a named-waterbody match might help beyond terrain and distance. Blanking names at training and test changed riverine MAE from **4.8149 to 4.8153 ft** across 13 archived floods. The paired 95% interval for the benefit was **−0.034 to +0.034 ft**. The study found no measurable improvement from those tags. A shared name is only a proxy for connectivity between channel reaches. [Waterbody experiment protocol](WATERBODY_PROTOCOL.md) · [Waterbody results](waterbody_ablation_results.json) · [USGS stream-network data](https://www.usgs.gov/national-hydrography/nhdplus-high-resolution)

## What to measure next

An extra reading only helps if it improves estimates at the sites that are still unmeasured. In a separate replay of the 13 floods, we split flags into starting measurements, candidates for an extra survey, and a permanently hidden scoring set. The deployed model was refit on the original training floods. All policies then reused that frozen fit and received the same number of new readings.

| Extra budget per eligible environment | No added readings (ft) | Random (ft) | Terrain coverage (ft) | Conditional variance (ft) |
| --- | ---: | ---: | ---: | ---: |
| One mark | 4.40 | 4.34 | 4.269 | 4.272 |
| About 10% of flags | 4.40 | 4.03 | 3.67 | 3.69 |

The research policy uses a terrain and waterbody Matérn kernel. It greedily chooses the mark whose measurement would reduce the sum of conditional variances most across the hidden locations. Each candidate’s score is the sum of squared conditional covariances with those locations, divided by the candidate’s conditional variance plus observation noise. A direct covariance recalculation agreed with the rank-one implementation in 81 archived cases and one synthetic case.

The variance policy failed the predeclared superiority threshold. For one extra mark, its paired advantage over terrain coverage was **−0.003 ft**, with a 95% interval of **−0.045 to +0.041 ft**. Terrain coverage had lower observed error than random and farthest-first at both budgets. The released map keeps its existing review flags, which indicate disagreement about the current estimate. This experiment measures retrospective placement decisions and leaves travel effort and access unmodeled. [Protocol](SURVEY_PROTOCOL.md) · [Development lock](survey_selection.json) · [Full results](survey_decision_results.json) · [Spatial observation design research](https://jmlr.org/papers/v9/krause08a.html)

## Model and benchmark

The model starts from a physical hypothesis. On the coast the nearby water surface provides the starting estimate. On rivers the nearby rise above terrain is a starting estimate, adjusted for differences in ground elevation. Histogram gradient boosting learns when to correct that estimate based on distance between marks, terrain shape and whether the waterbody matches. The fit uses Sandy, Matthew, Harvey and Irma. Florence and Michael select the method and its strongest transparent comparator. Ian and Helene hold out whole storms. Four site IDs recur across the older and held-out storms. The geographic split measures transfer to later events with some observations already available from them.

We tested nearest observation, inverse-distance weighting, transfer of nearby height above terrain, bounded local terrain regression and a PyTorch attention network. The neural model mixed the observed water surface with terrain-adjusted observations. It had **6.23 ft** pooled error, largely due to steep terrain and sparse measurements in Helene, and was rejected. Helene’s simple terrain-plus-depth method was close to the winning model at **6.72 ft**. The comparison baseline is fixed by development performance across methods. [Development results](selection.json) · [Benchmark design](PROTOCOL.md)

The main test starts with about 15% of flags in each environment, using at least two observations where possible. Predictions can read elevations only from those revealed observations. The study keeps one approved mark per site and one flag per small geographic cell within each storm. The 10 km spatial bootstrap resamples blocks within events, preserving all repetitions of a site.

An exploratory blend rechecked each observed site using its other survey readings, then used local reconstruction error to weight three estimates. Its MAE rose from 2.08 to 2.10 ft on the original holdout and remained near 5.30 ft on the additional events. In Helene the blend reduced the share of errors above five feet from 49.1% to 48.0%, while increasing average error from 6.65 to 6.77 ft. The release uses the frozen boosted model. [Adaptive weighting experiment](adaptive_results.json)

The distinct contribution is a reproducible evaluation of the partial survey and an offline tool that recomputes both estimates as a new measurement arrives. Existing work has used high-water observations for flood mapping and interpolated river surfaces from gauges. Prior ML studies also use terrain and high-water marks for flood-depth prediction. [USGS flood mapping study](https://pubs.usgs.gov/sir/2018/5070/) · [Search-constrained inverse-distance weighting](https://doi.org/10.1007/s11069-020-04309-w) · [ML flood-depth study](https://doi.org/10.1029/2024WR039244) · [Terrain and weather fusion preprint](https://arxiv.org/abs/2611.05220)

## Reproduce

Python 3.12 on macOS ARM64 was used for the reported run. Once downloaded, the browser demo runs offline.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python map_model.py validation
python map_model.py test
python external_eval.py evaluate
python sensitivity.py
python review.py
python kriging.py validation
python kriging.py test
python replay_remote.py evaluate
python replay_remote.py kriging
python survey_decision.py validation
python survey_decision.py test
python waterbody_ablation.py
python terrain_ablation.py
python demo_export.py
```

These commands use the supplied processed [survey panel](panel.csv) and regenerate the validation selection, holdout metrics and browser demo. They fit fresh models. For tracing the upstream inputs, `python fetch_events.py` pulls the USGS event data and `python terrain.py` joins the public ground-elevation tiles. For the separate three-event benchmark, `python external_eval.py prepare` rebuilds its processed panel from the current catalog. Upstream services can revise the data. The supplied panels are the benchmark snapshots.

## Regression tests

With Node.js 18 or later on your PATH, run `python -m unittest -v test_markroute` from the repository root. The suite checks that the event splits and manifests agree, that hidden elevations cannot influence model inputs, and that a review prediction excludes its own observed elevation. It refits the frozen model, reproduces archived errors on all 13 evaluation floods, and compares thousands of predictions from the offline JavaScript engine to Python before and after adding an archived survey. It checks the signed survey contributions against a separate enumeration of insertion orders.

## Scope and data

Results apply to already flagged archive locations and simulated availability of survey measurements. MarkRoute estimates peak water-surface elevation above the NAVD88 reference. The estimate at a flag cannot establish flood extent, travel safety or where to conduct a new survey in practice. Terrain is a separate raster product at finite resolution, and its approximate values are features. Waterbody and environment metadata are assumed to be available at flagging in the replay. Site access, route effort, changing flood conditions and unsuccessful surveys are outside the evaluation.

- Survey data and methods. [USGS Flood Event Viewer](https://www.usgs.gov/tools/flood-event-viewer) · [High-water mark field guide](https://pubs.usgs.gov/tm/03/a24/tm3a24.pdf)
- Independent ground elevation. [Mapzen Terrain Tiles](https://www.mapzen.com/blog/terrain-tile-service/) · [Source attribution](https://github.com/tilezen/joerd/blob/master/docs/attribution.md)
- Model and evaluation libraries. [scikit-learn](https://scikit-learn.org/) · [PyTorch](https://pytorch.org/)
- AI assistance. [OpenAI Codex](https://openai.com/codex/) assisted with research, implementation, testing and drafting.
