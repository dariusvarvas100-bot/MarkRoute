"""Benchmark the value of extra survey readings on permanently hidden sites."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from kriging import matern32
from map_model import DEV, SEED, TEST, TRAIN, fit_boosted, observation_features, sample, split_points, xy_km

HERE = Path(__file__).resolve().parent
BUDGETS = ("one", "tenPercent")
CONTROLS = ("random", "farthest", "terrain coverage", "disagreement")
TERRAIN_DESIGNS = {"terrain IVR": 0.75, "terrain IVR wide": 2.5}
GEOGRAPHIC = "geographic IVR"
NO_EXTRA = "no extra"
METHODS = (NO_EXTRA, *CONTROLS, GEOGRAPHIC, *TERRAIN_DESIGNS)
NUGGET = 0.25
MASKS = range(SEED, SEED + 12)


def partition(frame, seed):
    initial, unknown = split_points(frame, seed)
    rng = np.random.default_rng(seed + 2481)
    for ids in frame.groupby("environment").indices.values():
        hidden = rng.permutation(np.intersect1d(unknown, ids))
        anchors = np.intersect1d(initial, ids)
        if len(hidden) >= 4 and len(anchors):
            candidates, scoring = np.split(hidden, [len(hidden) // 2])
            yield anchors, candidates, scoring, min(len(candidates), max(1, round(len(ids) / 10)))


def kernel(frame, distances, anchors, hidden, factor, terrain=True):
    nearest = distances[np.ix_(hidden, anchors)].min(axis=1)
    length = max(0.25, factor * np.median(nearest))
    scaled = distances / length
    if terrain:
        ground = frame.terrain_median_ft.to_numpy()
        scaled = np.hypot(scaled, (ground[:, None] - ground[None, :]) / 45)
        water = frame.waterbody.to_numpy()
        shared = (water[:, None] == water[None, :]) & (water[:, None] != "")
        correlation = matern32(scaled) * (0.35 + 0.65 * shared)
        np.fill_diagonal(correlation, 1)
        return correlation
    return matern32(scaled)


def variance_reduction(covariance, anchors, candidates, scoring, count):
    hidden = np.r_[candidates, scoring]
    cross = covariance[np.ix_(hidden, anchors)]
    measured = covariance[np.ix_(anchors, anchors)] + NUGGET * np.eye(len(anchors))
    conditional = covariance[np.ix_(hidden, hidden)] - cross @ cho_solve(cho_factor(measured), cross.T)
    active = np.arange(len(candidates))
    chosen = []
    for _ in range(count):
        variance = conditional[active, active] + NUGGET
        benefit = np.square(conditional[len(candidates):, active]).sum(axis=0) / variance
        position = int(np.argmax(benefit))
        index = active[position]
        chosen.append(candidates[index])
        influence = conditional[:, index]
        conditional -= np.outer(influence, influence) / variance[position]
        active = np.delete(active, position)
    return np.asarray(chosen)


def greedy_coverage(distances, terrain_kernel, anchors, candidates, scoring, count):
    nearest = distances[np.ix_(candidates, anchors)].min(axis=1)
    current = terrain_kernel[np.ix_(scoring, anchors)].max(axis=1)
    candidate_cover = terrain_kernel[np.ix_(scoring, candidates)]
    farthest, terrain = [], []
    for _ in range(count):
        index = int(np.argmax(nearest))
        farthest.append(candidates[index])
        nearest = np.minimum(nearest, distances[candidates, candidates[index]])
        nearest[index] = -np.inf

        gains = np.maximum(0., candidate_cover - current[:, None]).sum(axis=0)
        gains[terrain] = -np.inf
        index = int(np.argmax(gains))
        terrain.append(index)
        current = np.maximum(current, candidate_cover[:, index])
    return np.asarray(farthest), candidates[terrain]


def rank_measurements(model, frame, xy, distances, anchors, candidates, scoring, count, seed):
    """Frame contains elevations only for the initial anchors."""
    assert frame.elev_ft.notna().sum() == len(anchors)
    hidden = np.r_[candidates, scoring]
    candidate_data = observation_features(frame, xy, distances, anchors, candidates)
    predicted = model.predict(candidate_data["features"]) + candidate_data["default"]
    disagreement = np.abs(predicted - candidate_data["baseline"][:, 3])
    terrain = kernel(frame, distances, anchors, hidden, TERRAIN_DESIGNS["terrain IVR"])
    farthest, coverage = greedy_coverage(distances, terrain, anchors, candidates, scoring, count)
    ranks = {
        NO_EXTRA: np.array([], dtype=int),
        "random": np.random.default_rng(seed + 5435).permutation(candidates)[:count],
        "farthest": farthest,
        "terrain coverage": coverage,
        "disagreement": candidates[np.argsort(-disagreement, kind="stable")[:count]],
        GEOGRAPHIC: variance_reduction(kernel(frame, distances, anchors, hidden, 0.75, terrain=False),
                                      anchors, candidates, scoring, count),
        "terrain IVR": variance_reduction(terrain, anchors, candidates, scoring, count),
    }
    for method, factor in TERRAIN_DESIGNS.items():
        if method not in ranks:
            ranks[method] = variance_reduction(kernel(frame, distances, anchors, hidden, factor),
                                               anchors, candidates, scoring, count)
    return ranks


def replay(model, points):
    frame = points.reset_index(drop=True)
    xy = xy_km(frame)
    distances = cdist(xy, xy)
    actual = frame.elev_ft.to_numpy()
    rows = []
    for seed in MASKS:
        for anchors, candidates, scoring, count in partition(frame, seed):
            visible = frame.assign(elev_ft=np.nan)
            visible.loc[anchors, "elev_ft"] = actual[anchors]
            ranks = rank_measurements(model, visible, xy, distances, anchors, candidates, scoring, count, seed)
            common = {
                "event": frame.event_name.to_numpy()[scoring],
                "environment": frame.environment.to_numpy()[scoring],
                "site": frame.site_id.to_numpy()[scoring],
                "block": [f"{int(x//10)}_{int(y//10)}" for x, y in xy[scoring]],
                "mask": seed,
                "actual": actual[scoring],
            }
            cache = {}
            for budget, limit in zip(BUDGETS, (1, count)):
                metrics = pd.DataFrame(common).assign(budget=budget)
                for method, ranking in ranks.items():
                    selected = ranking[:limit]
                    key = tuple(sorted(selected))
                    if key not in cache:
                        updated = visible.copy()
                        updated.loc[selected, "elev_ft"] = actual[selected]
                        measured = np.sort(np.r_[anchors, selected])
                        assert np.intersect1d(measured, scoring).size == 0
                        features = observation_features(updated, xy, distances, measured, scoring)
                        assert np.isnan(features["actual"]).all()
                        cache[key] = model.predict(features["features"]) + features["default"]
                    metrics[method] = np.abs(actual[scoring] - cache[key])
                rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def scores(frame):
    return {method: {"maeFt": float(frame[method].mean()),
                     "missOver5Ft": float(frame[method].gt(5).mean())} for method in METHODS}


def paired_interval(frame, policy, baseline):
    rng = np.random.default_rng(SEED)
    sums, counts = [], []
    for _, event in frame.groupby("event"):
        groups = event.assign(gain=event[baseline] - event[policy]).groupby("block").gain.agg(["sum", "size"])
        draw = rng.integers(len(groups), size=(4000, len(groups)))
        sums.append(groups["sum"].to_numpy()[draw].sum(axis=1))
        counts.append(groups["size"].to_numpy()[draw].sum(axis=1))
    simulations = np.arange(4000)[:, None]
    storms = rng.integers(len(sums), size=(4000, len(sums)))
    return np.quantile(np.asarray(sums).T[simulations, storms].sum(axis=1) /
                       np.asarray(counts).T[simulations, storms].sum(axis=1), [0.025, 0.975]).tolist()


def report(frame, policy, baseline):
    errors = scores(frame)
    events = {name: scores(value) for name, value in frame.groupby("event")}
    gain = errors[baseline]["maeFt"] - errors[policy]["maeFt"]
    interval = paired_interval(frame, policy, baseline)
    wins = sum(values[policy]["maeFt"] < values[baseline]["maeFt"] for values in events.values())
    relative = gain / errors[baseline]["maeFt"]
    return {
        "scoredEpisodes": len(frame), "uniqueScoredSites": int(frame.site.nunique()),
        "policy": policy, "sameBudgetControl": baseline, "gainFt": gain, "relativeGain": relative,
        "paired95IntervalFt": interval, "eventWins": wins,
        "passesHeadlineGate": bool(relative >= 0.1 and interval[0] > 0 and wins >= 9),
        "methods": errors, "events": events,
        "environments": {name: scores(value) for name, value in frame.groupby("environment")},
        "gainVsNoExtraFt": errors[NO_EXTRA]["maeFt"] - errors[policy]["maeFt"],
        "gainVsNoExtra95IntervalFt": paired_interval(frame, policy, NO_EXTRA),
    }


def fingerprint():
    files = ("SURVEY_PROTOCOL.md", "survey_decision.py", "map_model.py", "kriging.py", "panel.csv")
    return {name: sha256((HERE / name).read_bytes()).hexdigest() for name in files}


def evaluate(model, panel):
    replays = []
    for _, event in panel.groupby("event_id"):
        print(f"Replaying {event.event_name.iloc[0]}", flush=True)
        replays.append(replay(model, event))
    return pd.concat(replays, ignore_index=True)


def run(stage):
    original = pd.read_csv(HERE / "panel.csv", keep_default_na=False)
    lock_path = HERE / "survey_selection.json"
    with threadpool_limits(limits=4):
        model = fit_boosted(sample(original, TRAIN, 18))
        if stage == "validation":
            frame = evaluate(model, original[original.event_id.isin(DEV)])
            errors = {budget: scores(frame[frame.budget.eq(budget)]) for budget in BUDGETS}
            selection = {}
            for budget in BUDGETS:
                mae = {method: values["maeFt"] for method, values in errors[budget].items()}
                selection[budget] = {"policy": min(TERRAIN_DESIGNS, key=mae.get),
                                     "control": min(CONTROLS, key=mae.get)}
            result = {"freeze": fingerprint(), "selected": selection, "development": errors}
            output = lock_path
        else:
            locked = json.loads(lock_path.read_text())
            if locked["freeze"] != fingerprint():
                raise ValueError("Development lock does not match current inputs")
            external = pd.read_csv(HERE / "external_panel.csv", keep_default_na=False)
            remote = pd.read_csv(HERE / "remote_panel.csv", keep_default_na=False)
            frame = evaluate(model, pd.concat([original[original.event_id.isin(TEST)], external, remote]))
            result = {}
            for budget, methods in locked["selected"].items():
                result[budget] = report(frame[frame.budget.eq(budget)], methods["policy"], methods["control"])
            output = HERE / "survey_decision_results.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "test"))
    run(parser.parse_args().stage)
