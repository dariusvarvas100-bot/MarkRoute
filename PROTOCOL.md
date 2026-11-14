# MarkRoute benchmark protocol

A flood survey coordinator has flagged locations where physical evidence
of high water was visible. Some locations have surveyed elevations. Replay
the task of estimating peak water-surface elevation at the other flagged
sites and deciding where one further elevation measurement is most useful.
Use real USGS observations, with measurements hidden to simulate a partial
survey. This is retrospective and conditional on the flagged locations,
and estimates neither inundation extent nor safe routes. Each added reading
costs one unit. Travel costs and unsuccessful visits are outside this test.

Use approved datum NAVD88 and USGS excellent, good or fair marks. Keep one
observation per USGS site, favoring higher documented quality and then
stable ID. Also retain just one site per small spatial grid cell. Use
independent Mapzen ground elevations, local relief, distances, and the
reported waterbody/environment as features. Never give the network any
height-above-ground field reported by a surveyor. Treat waterbody fields as
available at flagging time in this retrospective experiment.

Train on Sandy, Matthew, Harvey and Irma. Select models on Florence and
Michael. Score Ian and Helene once. Keep every initial revealed mask
identical between methods, using fixed seeds. The initial budget is 15%
per event and environment with a minimum of two anchors where possible.
Report mean absolute water-surface error in feet for each event and
environment, plus the number of misses greater than 5 feet. Test against
nearest observed water surface, geographic interpolation, terrain plus
nearby median depth, a physically bounded local regression and learned
gradient boosting at the same information budget. Use PyTorch neighbor
attention to learn a continuous adjustment from surveyed water values
and terrain, learned only on older events. For a project-level prediction
claim require at least 10% less MAE than the validation-best comparator
across both held-out events, with no event worse by more than 5%.

The secondary acquisition experiment uses a three-way partition into
initial observations, allowable candidate flags and independent scoring
flags. Plan the same fixed extra survey allowance in each method and reveal
only the requested candidate measurements. Compare a spatially diverse
selection, explicit integrated influence, predicted residual uncertainty
and learned query selection if the latter is fitted. Report the realized
difference in error on the shared scoring flags, and whether rankings
remain useful after removing extremely steep terrain. Treat resampling
intervals as conditional on the archived events.

Before opening Ian and Helene, record the strongest numeric method on
the development storms regardless of model family. The headline practical
comparison separately tests that development-selected mapping method against
the strongest transparent nonlearned interpolation policy chosen on
development. Keep the attention-versus-boosting endpoint above regardless
of outcome. A claim that the survey selector improves decisions still
requires a test gain over the strongest development-ranked survey policy.
