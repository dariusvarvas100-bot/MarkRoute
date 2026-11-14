# MarkRoute

**AI + Climate · Veritas AI International High School Hackathon 2026**

After a flood, a survey coordinator has a list of locations where crews flagged physical marks left by high water. Only some flags have surveyed elevations yet. MarkRoute combines those first measurements with independent terrain data to estimate the peak water-surface elevation at the other flagged sites while surveys continue.

The method cut average error from **2.48 to 2.08 feet (16.0%)** against a terrain-aware baseline across two held-out storms, with the same surveyed marks supplied to each method. A separate catalog-selected test found **14.1% lower average error** on three more floods. All five events improved. More surveyed events and a prospective study would be needed to establish field performance.

[One-page write-up](MarkRoute_Writeup.pdf) · [106-second walkthrough](MarkRoute_Demo.mp4) · [Demo and judging notes](MarkRoute_Judging_Notes.md)

## Try a survey replay

Download the repo and open [demo.html](demo.html). It runs offline, including model inference.

1. Choose **Ian** or **Helene**. Blue observations are the initial surveys. The other points are locations with a recorded USGS mark whose elevation is hidden for the replay.
2. Select an unmeasured point to see how the two methods estimate its peak water-surface elevation. Reveal the archived value to see their actual errors.
3. Add the point to the survey and watch both methods recompute the remaining estimates. The display is a retrospective experiment using USGS marks at known flags.

A replay represents one fixed initial sample. The benchmark combines 16 initial samples per storm. As an interactive replay adds observations it changes which points remain unmeasured, and it reports scenario error separately.

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

## Model and benchmark

The model starts from a physical hypothesis. On the coast the nearby water surface provides the starting estimate. On rivers the nearby rise above terrain is a starting estimate, adjusted for differences in ground elevation. Histogram gradient boosting learns when to correct that estimate based on distance between marks, terrain shape and whether the waterbody matches. The fit uses Sandy, Matthew, Harvey and Irma. Florence and Michael select the method and its strongest transparent comparator. Ian and Helene hold out whole storms. Four site IDs recur across the older and held-out storms. The geographic split measures transfer to later events with some observations already available from them.

We tested nearest observation, inverse-distance weighting, transfer of nearby height above terrain, bounded local terrain regression and a PyTorch attention network. The neural model mixed the observed water surface with terrain-adjusted observations. It had **6.23 ft** pooled error, largely due to steep terrain and sparse measurements in Helene, and was rejected. Helene’s simple terrain-plus-depth method was close to the winning model at **6.72 ft**. The comparison baseline is fixed by development performance across methods. [Development results](selection.json) · [Benchmark design](PROTOCOL.md)

The main test starts with about 15% of flags in each environment, using at least two observations where possible. Predictions can read elevations only from those revealed observations. The study keeps one approved mark per site and one flag per small geographic cell within each storm. The 10 km spatial bootstrap resamples blocks within events, preserving all repetitions of a site.

The distinct contribution is a reproducible evaluation of the partial survey and an offline tool that recomputes both estimates as a new measurement arrives. Existing work has used high-water observations for flood mapping and interpolated river surfaces from gauges. Prior ML studies also use terrain and high-water marks for flood-depth prediction. [USGS flood mapping study](https://pubs.usgs.gov/sir/2018/5070/) · [Search-constrained inverse-distance weighting](https://doi.org/10.1007/s11069-020-04309-w) · [ML flood-depth study](https://doi.org/10.1029/2024WR039244) · [Terrain and weather fusion preprint](https://arxiv.org/abs/2611.05220)

## Reproduce

Python 3.12 on macOS ARM64 was used for the reported run. The browser demo needs no Python or network.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python map_model.py validation
python map_model.py test
python demo_export.py
python external_eval.py evaluate
```

These commands use the supplied processed [survey panel](panel.csv) and regenerate the validation selection, holdout metrics and browser demo. They fit fresh models. For tracing the upstream inputs, `python fetch_events.py` pulls the USGS event data and `python terrain.py` joins the public ground-elevation tiles. For the separate three-event benchmark, `python external_eval.py prepare` rebuilds its processed panel from the current catalog. Upstream services can revise the data. The supplied panels are the benchmark snapshots.

## Scope and data

Results apply to already flagged archive locations and simulated availability of survey measurements. MarkRoute estimates peak water-surface elevation above the NAVD88 reference. The estimate at a flag cannot establish flood extent, travel safety or where to conduct a new survey in practice. Terrain is a separate raster product at finite resolution, and its approximate values are features. Waterbody and environment metadata are assumed to be available at flagging in the replay. Site access, route effort, changing flood conditions and unsuccessful surveys are outside the evaluation.

- Survey data and methods. [USGS Flood Event Viewer](https://www.usgs.gov/tools/flood-event-viewer) · [High-water mark field guide](https://pubs.usgs.gov/tm/03/a24/tm3a24.pdf)
- Independent ground elevation. [Mapzen Terrain Tiles](https://www.mapzen.com/blog/terrain-tile-service/) · [Source attribution](https://github.com/tilezen/joerd/blob/master/docs/attribution.md)
- Model and evaluation libraries. [scikit-learn](https://scikit-learn.org/) · [PyTorch](https://pytorch.org/)
- AI assistance. [OpenAI Codex](https://openai.com/codex/) assisted with research, implementation, testing and drafting.
