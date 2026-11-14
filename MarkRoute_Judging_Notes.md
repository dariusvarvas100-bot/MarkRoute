# MarkRoute judging notes

## Two-minute demo

- **0:00–0:18.** “After a flood, a crew can flag a debris line before it has time to measure its elevation. MarkRoute helps a survey coordinator make a provisional estimate at already-flagged sites with the measurements they have.”
- **0:18–0:55.** Open Ian, Survey 1. Show the 44 starting measurements. On mark 32828, click **Reveal this mark**. “The crew measured 12.30 feet. MarkRoute estimated 12.51 and bounded regression estimated 10.45.” Click **Add measurement**. “Both models now use that reading and recompute on the remaining marks.”
- **0:55–1:25.** “Where terrain changes quickly, the nearest water level can be a poor starting point. Boosted trees correct a starting estimate based on nearby water elevations and local terrain. In Ian and Helene, average error fell from 2.48 to 2.08 feet. In a separate three-flood check, it fell from 6.17 to 5.30 feet. All five events improved.”
- **1:25–1:52.** Switch to Helene and select mark 35849. “This mark is an example where the baseline was closer. Nearly half of all Helene estimates still missed by over five feet. The next step would be to replay the survey with a local crew using real travel time and site access.”

If you have time, show the added-survey control so judges can see both estimates recompute with the same new measurement. Keep the main result on the fixed initial-survey comparison. The separate adaptive-survey experiment used the attention model.

## Questions to rehearse

- **What is the original contribution?** USGS already uses high-water marks to interpolate flood surfaces. Prior teams have also trained models on those marks. MarkRoute isolates a different question, about how accurately a coordinator can estimate the elevations of unsurveyed flagged marks when several same-storm anchors are already available. It evaluates that exact workflow at the same survey budget against a bounded terrain-aware rule. [USGS Harvey mapping](https://pubs.usgs.gov/sir/2018/5070/), [Pakdehi et al.](https://doi.org/10.1029/2024WR039244), [Zhou et al., preprint](https://arxiv.org/abs/2611.05220)
- **What did the model get to see?** Existing survey elevations from the same storm at a seeded, roughly 15% initial budget by environment, plus site locations, terrain and recorded waterbody/environment. The historical replay assumes those fields are available when surveyors flag a site. It hides test elevations at other sites. Training and model choice used six earlier storms.
- **Why use boosting when you built attention?** Validation on Florence and Michael selected boosting before opening Ian and Helene. The attention model missed by an average of 29.5 ft in Helene. The original holdout had 355 sites and 4,816 scored predictions. The catalog follow-up added Ida, July 2025 Texas and December 2025 Pacific Northwest, with 659 sites and 8,960 scored predictions. Repeated masks reuse sites. The paired spatial intervals summarize these events.
- **What exactly did you measure?** Peak water-surface elevation at a physical mark, in feet relative to NAVD88. MAE is the average size of the mistake. The demo covers archived, already-flagged locations after a flood. Any proposed operational use would need field validation with access, travel, and finer terrain data.
- **What does the equal-budget survey result establish?** It is an exploratory comparison using the attention estimator. Extra measurement counts are identical, with error scored on a third group of marks outside the survey queue. The 16.0 percent number is for the released boosted estimator on the original initial-estimation experiment. Travel time and unsuccessful visits require a field test.

For live finals, take the extra time to show one Helene miss and explain one MAE calculation in the interface. Credit the USGS records, Mapzen terrain tiles, scikit-learn, PyTorch and Codex. Use your own explanation of the replay and how the flags were hidden.
