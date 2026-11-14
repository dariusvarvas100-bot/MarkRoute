"""Compare learning formulations with identical terrain and survey features."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from map_model import DEV, STATIC_NAMES, TRAIN, fit_boosted, sample
from waterbody_ablation import FULL, paired_interval

HERE = Path(__file__).resolve().parent
DIRECT = "direct elevation"
DISTANCE = "distance-weighted residual"
BOUNDED = "bounded-terrain residual"
DEPLOYED = "released MarkRoute"
ALTERNATIVES = (DIRECT, DISTANCE, BOUNDED)
STRATEGIES = (FULL, *ALTERNATIVES, DEPLOYED)
DISTANCE_INDEX = STATIC_NAMES.index("geographic interpolation")
BOUNDED_INDEX = STATIC_NAMES.index("bounded local regression")


def comparisons(data):
    start = data["default"]
    idw = data["baseline"][:, DISTANCE_INDEX]
    bounded = data["baseline"][:, BOUNDED_INDEX]
    features = np.column_stack((data["features"], start, idw, bounded))
    offsets = {FULL: start, DIRECT: np.zeros_like(start), DISTANCE: idw, BOUNDED: bounded}
    return features, offsets


def fit_choices(training):
    features, offsets = comparisons(training)
    models = {name: fit_boosted({"features": features, "actual": training["actual"], "default": offset})
              for name, offset in offsets.items()}
    models[DEPLOYED] = fit_boosted(training)
    return models


def errors_for(data, models):
    features, offsets = comparisons(data)
    rows = pd.DataFrame({key: data[key] for key in ("site", "event", "block", "environment")})
    for name, model in models.items():
        values, offset = (data["features"], data["default"]) if name == DEPLOYED else (features, offsets[name])
        rows[name] = np.abs(data["actual"] - (model.predict(values) + offset))
    return rows


def describe(rows):
    return {name: {"maeFt": float(rows[name].mean()), "shareOverFiveFt": float(rows[name].gt(5).mean())}
            for name in STRATEGIES}


def report(rows, reference):
    methods = describe(rows)
    gain = methods[reference]["maeFt"] - methods[FULL]["maeFt"]
    interval = paired_interval(rows, ablated=reference)
    relative = gain / methods[reference]["maeFt"]
    by_event = {label: describe(event) for label, event in rows.groupby("event")}
    return {
        "sites": int(rows.site.nunique()),
        "scoredMaskSitePairs": int(len(rows)),
        "methods": methods,
        "eventResults": by_event,
        "eventWins": sum(event[FULL]["maeFt"] < event[reference]["maeFt"] for event in by_event.values()),
        "relativeGain": relative,
        "paired95IntervalFt": interval,
        "meetsThreshold": bool(relative >= 0.05 and interval[0] > 0),
    }


def fingerprint():
    files = ("RESIDUAL_PROTOCOL.md", "residual_formulation.py", "map_model.py", "waterbody_ablation.py",
             "panel.csv", "remote_panel.csv")
    return {name: sha256((HERE / name).read_bytes()).hexdigest() for name in files}


def run(stage):
    archive = pd.read_csv(HERE / "panel.csv", keep_default_na=False)
    lock_path = HERE / "residual_selection.json"
    with threadpool_limits(4):
        models = fit_choices(sample(archive, TRAIN, 18))
        if stage == "validation":
            dev = errors_for(sample(archive, DEV, 16), models)
            stats = describe(dev[dev.environment.eq("Riverine")])
            reference = min(ALTERNATIVES, key=lambda name: stats[name]["maeFt"])
            result = {"freeze": fingerprint(), "selectedAlternative": reference, "developmentRiverine": stats}
            output = lock_path
        else:
            locked = json.loads(lock_path.read_text())
            if locked["freeze"] != fingerprint():
                raise ValueError("Development lock does not match the current experiment")
            reference = locked["selectedAlternative"]
            if reference not in ALTERNATIVES:
                raise ValueError("Invalid comparison strategy")
            remote = pd.read_csv(HERE / "remote_panel.csv", keep_default_na=False)
            replay = errors_for(sample(remote, remote.event_id.unique(), 16), models)
            result = {"frozenAlternative": reference,
                      "primaryRiverine": report(replay[replay.environment.eq("Riverine")], reference),
                      "secondaryAll": report(replay, reference)}
            output = HERE / "residual_formulation_results.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "test"))
    run(parser.parse_args().stage)
