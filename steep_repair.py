"""Test loss functions and local weighting for large-error flood estimates."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from map_model import DEV, TEST, TRAIN, fit_boosted, sample
from waterbody_ablation import FULL, paired_interval

ROOT = Path(__file__).resolve().parent
QUANTILES = np.array([.05, .15, .3, .5, .7, .85, .95])
BASELINE = "published"
WINDOW = "modal window"
ABSOLUTE = "weighted absolute"
SQUARED = "weighted squared"
VARIANTS = {WINDOW: (.25, .5, 1.), ABSOLUTE: (.5, 1.), SQUARED: (.5, 1.)}
ARCHIVES = ("external_panel.csv", "remote_panel.csv", "archival_panel.csv")
CUTOFF = 35.
TOLERANCE = 5.


def is_steep(data):
    return (data["environment"] == "Riverine") & (data["pair"][:, 0, 5] >= np.log1p(CUTOFF))


def fit_alternative(training, loss, alpha=None, weighted=False):
    original = fit_boosted if loss == "absolute_error" and not weighted else None
    if original:
        return original(training)
    settings = {"loss": loss, "learning_rate": .08, "max_iter": 180, "max_leaf_nodes": 18,
                "l2_regularization": 1., "random_state": 7162}
    if alpha is not None:
        settings["quantile"] = alpha
    weight = np.where(is_steep(training), 3., 1.) if weighted else None
    model = HistGradientBoostingRegressor(**settings)
    return model.fit(training["features"], training["actual"] - training["default"], sample_weight=weight)


def modal_center(estimated, reference):
    knots = np.sort(estimated, axis=1)
    # Extend the outermost slope to 0 and 1 to keep probability mass continuous.
    left = knots[:, 0] - np.maximum(.01, knots[:, 1] - knots[:, 0]) / 2
    right = knots[:, -1] + np.maximum(.01, knots[:, -1] - knots[:, -2]) / 2
    knots = np.column_stack((left, knots, right))
    probabilities = np.diff(np.r_[0., QUANTILES, 1.])
    starts = knots[:, None, :-1]
    lengths = np.maximum(1e-8, np.diff(knots, axis=1))[:, None, :]

    def cdf(position):
        coverage = np.clip((position[:, :, None] - starts) / lengths, 0., 1.)
        return coverage @ probabilities

    centers = np.column_stack((knots - TOLERANCE, knots + TOLERANCE, reference))
    mass = cdf(centers + TOLERANCE) - cdf(centers - TOLERANCE)
    largest = mass.max(axis=1, keepdims=True)
    distance = np.where(np.abs(mass - largest) < 1e-10, np.abs(centers - reference[:, None]), np.inf)
    return centers[np.arange(len(centers)), distance.argmin(axis=1)]


def fit(training, methods):
    trained = {BASELINE: fit_boosted(training)}
    for method in methods:
        if method == WINDOW:
            trained[WINDOW] = [fit_alternative(training, "quantile", alpha) for alpha in QUANTILES]
        else:
            loss = "absolute_error" if method == ABSOLUTE else "squared_error"
            trained[method] = fit_alternative(training, loss, weighted=True)
    return trained


def results(data, models, variants):
    steep = is_steep(data)
    actual = data["actual"][steep]
    estimates = data["default"][steep]
    features = data["features"][steep]
    records = pd.DataFrame({key: data[key][steep] for key in ("event", "site", "block")})
    base_residual = models[BASELINE].predict(features)
    records[BASELINE] = np.abs(actual - (estimates + base_residual))
    for method, weights in variants.items():
        if method == WINDOW:
            quantiles = np.column_stack([model.predict(features) for model in models[WINDOW]])
            residual = modal_center(quantiles, base_residual)
        else:
            residual = models[method].predict(features)
        for weight in weights:
            prediction = estimates + (1. - weight) * base_residual + weight * residual
            records[f"{method} × {weight:g}"] = np.abs(actual - prediction)
    return records


def summarize(frame, method):
    return {"maeFt": float(frame[method].mean()), "largeMissRate": float(frame[method].gt(TOLERANCE).mean()),
            "largeMisses": int(frame[method].gt(TOLERANCE).sum()),
            "pairs": len(frame), "uniqueEventSites": len(frame[["event", "site"]].drop_duplicates())}


def assess(frame, method, include_interval=False):
    baseline = summarize(frame, BASELINE)
    value = summarize(frame, method)
    gain = baseline["largeMissRate"] - value["largeMissRate"]
    event = {}
    for label, group in frame.groupby("event"):
        after, before = summarize(group, method), summarize(group, BASELINE)
        event[label] = {BASELINE: before, method: after,
                        "largeMissReduction": before["largeMissRate"] - after["largeMissRate"]}
    result = {BASELINE: baseline, method: value, "largeMissReduction": gain, "events": event}
    if include_interval:
        binary = frame[["event", "block"]].assign(**{
            FULL: frame[method].gt(TOLERANCE).astype(int), BASELINE: frame[BASELINE].gt(TOLERANCE).astype(int)})
        result["paired95Reduction"] = paired_interval(binary, ablated=BASELINE)
    return result


def fingerprint():
    files = ("STEEP_REPAIR_PROTOCOL.md", "steep_repair.py", "map_model.py", "panel.csv", *ARCHIVES)
    return {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def write(file, document):
    (ROOT / file).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


def run(stage, reserve=None):
    panel = pd.read_csv(ROOT / "panel.csv", keep_default_na=False)
    lock = ROOT / "steep_repair_selection.json"
    with threadpool_limits(4):
        training = sample(panel, TRAIN, 18)
        if stage == "validation":
            models = fit(training, VARIANTS)
            frames = results(sample(panel, DEV, 16), models, VARIANTS)
            records = {}
            eligible = []
            for method, weights in VARIANTS.items():
                for weight in weights:
                    name = f"{method} × {weight:g}"
                    outcome = assess(frames, name)
                    records[name] = outcome
                    if (outcome["largeMissReduction"] >= .015 and
                        outcome[name]["maeFt"] <= 1.02 * outcome[BASELINE]["maeFt"] and
                        min(x["largeMissReduction"] for x in outcome["events"].values()) >= -.02):
                        eligible.append((outcome["largeMissReduction"], method, weight))
            chosen = max(eligible) if eligible else None
            write(lock.name, {"freeze": fingerprint(), "selected": list(chosen[1:]) if chosen else None,
                              "development": records})
            print(f"Selected correction: {chosen}", flush=True)
            return
        selected = json.loads(lock.read_text())
        if selected["freeze"] != fingerprint():
            raise ValueError("The development lock has changed")
        if not selected["selected"]:
            print("No candidate passed the development gate.")
            return
        method, weight = selected["selected"]
        name = f"{method} × {weight:g}"
        model = fit(training, [method])
        original = panel[panel.event_id.isin(TEST)]
        groups = [original, *(pd.read_csv(ROOT / file, keep_default_na=False) for file in ARCHIVES)]
        archive = pd.concat(groups, ignore_index=True)
        source = pd.read_csv(reserve, keep_default_na=False) if reserve else archive
        if reserve and (set(source.event_id) & (set(archive.event_id) | set(panel.event_id))):
            raise ValueError("Reserved event ID already appears in the released archive or development data")
        replay = results(sample(source, source.event_id.unique(), 16), model, {method: (weight,)})
        outcome = assess(replay, name, include_interval=True)
        if reserve:
            outcome["reservedPanelSha256"] = sha256(Path(reserve).read_bytes()).hexdigest()
            outcome["meetsReserveGate"] = bool(outcome["largeMissReduction"] > 0 and
                                                outcome[name]["maeFt"] <= 1.02 * outcome[BASELINE]["maeFt"])
        else:
            outcome["meetsAdoptionGate"] = bool(
                outcome["largeMissReduction"] >= .01 and outcome["paired95Reduction"][0] > 0 and
                outcome[name]["maeFt"] <= 1.02 * outcome[BASELINE]["maeFt"] and
                all(event["largeMissReduction"] >= -.03 for event in outcome["events"].values()
                    if event[BASELINE]["uniqueEventSites"] >= 30))
        write("steep_repair_reserve_results.json" if reserve else "steep_repair_results.json", outcome)
        print(f"{name}: {outcome['largeMissReduction']:+.2%} fewer large misses", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "test"))
    parser.add_argument("--reserve", type=Path)
    args = parser.parse_args()
    run(args.stage, args.reserve)
