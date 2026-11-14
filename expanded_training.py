"""Test whether a broader archival fit reduces five-foot misses at steep sites."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from archival_replay import ARCHIVES as EARLIER, xyz
from map_model import TRAIN, fit_boosted, sample
from scipy.spatial import cKDTree
from sklearn.base import clone
from steep_repair import BASELINE, ROOT, assess, is_steep, write

KNOWN = ("panel.csv", "external_panel.csv", "remote_panel.csv", "archival_panel.csv")
VALIDATION_IDS = (125, 319, 346)
MODES = ("pooled", "equal event")
WEIGHTS = (0.5, 1.)
LOCK = "expanded_training_selection.json"


def fingerprint():
    files = ("EXPANDED_TRAINING_PROTOCOL.md", "expanded_training.py", "steep_repair.py",
             "map_model.py", "archival_replay.py", "waterbody_ablation.py", *KNOWN)
    return {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def fit_models(original, augmented, modes):
    frozen = fit_boosted(original)
    expanded = {}
    for mode in modes:
        if mode == "pooled":
            expanded[mode] = fit_boosted(augmented)
        else:
            labels, count = np.unique(augmented["event"], return_inverse=True, return_counts=True)[1:]
            weights = len(labels) / (len(count) * count[labels])
            expanded[mode] = clone(frozen).fit(augmented["features"],
                                               augmented["actual"] - augmented["default"], sample_weight=weights)
    return frozen, expanded


def replay(points, frozen, models, weights):
    data = sample(points, points.event_id.unique(), 16)
    keep = is_steep(data)
    actual, start, features = data["actual"][keep], data["default"][keep], data["features"][keep]
    base = start + frozen.predict(features)
    frame = pd.DataFrame({key: data[key][keep] for key in ("event", "block", "site")})
    frame[BASELINE] = np.abs(actual - base)
    for mode, model in models.items():
        estimate = start + model.predict(features)
        for weight in weights:
            frame[f"{mode} × {weight:g}"] = np.abs(actual - ((1 - weight) * base + weight * estimate))
    return frame


def passes(outcome, key, threshold, event_tolerance):
    return (outcome["largeMissReduction"] >= threshold and
            outcome[key]["maeFt"] <= 1.02 * outcome[BASELINE]["maeFt"] and
            all(value["largeMissReduction"] >= -event_tolerance for value in outcome["events"].values()))


def run(stage, reserve):
    known = [pd.read_csv(ROOT / name, keep_default_na=False) for name in KNOWN]
    original = known[0]
    archive = known[-1]
    extra = archive[archive.cohort.eq(EARLIER)].copy()
    training = pd.concat([original[original.event_id.isin(TRAIN)], extra], ignore_index=True)
    validation = pd.concat(known, ignore_index=True)
    validation = validation[validation.event_id.isin(VALIDATION_IDS)]
    assert set(training.event_id).isdisjoint(validation.event_id)
    with threadpool_limits(4):
        old = sample(original, TRAIN, 18)
        expanded = sample(training, training.event_id.unique(), 18)
        if stage == "validation":
            baseline, alternatives = fit_models(old, expanded, MODES)
            frame = replay(validation, baseline, alternatives, WEIGHTS)
            reports = {}
            acceptable = []
            for mode in MODES:
                for weight in WEIGHTS:
                    key = f"{mode} × {weight:g}"
                    outcome = assess(frame, key, include_interval=True)
                    reports[key] = outcome
                    if passes(outcome, key, .01, .03):
                        acceptable.append((outcome["largeMissReduction"], mode, weight))
            selected = max(acceptable) if acceptable else None
            write(LOCK, {"freeze": fingerprint(), "selected": list(selected[1:]) if selected else None,
                         "trainingEvents": int(training.event_id.nunique()),
                         "originalSteepSites": int(original[original.event_id.isin(TRAIN) &
                               original.environment.eq("Riverine") & original.relief_ft.ge(35)].site_id.nunique()),
                         "expandedSteepEventSites": int(len(training[training.environment.eq("Riverine") &
                                                                     training.relief_ft.ge(35)])),
                         "results": reports})
            print("Selected expanded fit:", selected, flush=True)
            return
        lock = json.loads((ROOT / LOCK).read_text())
        if lock["freeze"] != fingerprint():
            raise ValueError("The development lock has changed")
        if not lock["selected"]:
            print("No expanded fit qualified on development.")
            return
        mode, weight = lock["selected"]
        points = pd.read_csv(reserve, keep_default_na=False)
        prior = pd.concat(known)
        if set(points.event_id) & set(prior.event_id):
            raise ValueError("A reserved event is already in a published panel")
        baseline, alternatives = fit_models(old, expanded, (mode,))
        frame = replay(points, baseline, alternatives, (weight,))
        key = f"{mode} × {weight:g}"
        outcome = assess(frame, key, include_interval=True)
        chord = cKDTree(xyz(training)).query(xyz(points))[0]
        distances = 2 * 6371.0088 * np.arcsin(np.clip(chord / 2, 0, 1))
        outcome["distanceFromExpandedTraining"] = {"minimumKm": float(distances.min()),
                                                   "medianKm": float(np.median(distances)),
                                                   "withinOneKm": int((distances <= 1).sum())}
        outcome["reservedPanelSha256"] = sha256(reserve.read_bytes()).hexdigest()
        outcome["adoptionGate"] = bool(len(outcome["events"]) >= 2 and outcome[BASELINE]["uniqueEventSites"] >= 50
                                       and passes(outcome, key, .02, .03) and outcome["paired95Reduction"][0] > 0)
        write("expanded_training_reserve_results.json", outcome)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "reserve"))
    parser.add_argument("--panel", type=Path)
    args = parser.parse_args()
    if args.stage == "reserve" and args.panel is None:
        parser.error("Reserve stage requires a separate field panel.")
    run(args.stage, args.panel)
