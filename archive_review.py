"""Audit the tradeoff between reporting fewer estimates and making large errors."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from map_model import DEV, SEED, TRAIN, episode, fit_boosted, sample

HERE = Path(__file__).resolve().parent
FRACTIONS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
METHODS = ("disagreement", "relief", "nearest survey", "mean rank")


def scores(frame, model):
    for _, flood in frame.groupby("event_id"):
        for seed in range(SEED, SEED + 16):
            replay = episode(flood, seed)
            predictions = model.predict(replay["features"]) + replay["default"]
            error = np.abs(replay["actual"] - predictions)
            inland = replay["environment"] == "Riverine"
            if not inland.any():
                continue
            heuristics = np.column_stack((
                np.abs(predictions[inland] - replay["baseline"][inland, 3]),
                replay["pair"][inland, 0, 5],
                replay["pair"][inland, 0, 0],
            ))
            ranks = rankdata(heuristics, axis=0)
            priority = np.column_stack((heuristics, ranks.mean(axis=1)))
            yield str(replay["event"][inland][0]), error[inland], priority


def summaries(episodes, fraction, method):
    errors = {}
    total = {}
    for event, error, priority in episodes:
        budget = max(1, round(len(error) * fraction))
        order = np.argsort(priority[:, METHODS.index(method)], kind="stable")
        errors.setdefault(event, []).append(error[order[:budget]])
        total[event] = total.get(event, 0) + len(error)

    def summary(values, denominator):
        return {"reported": len(values), "available": denominator,
                "coverage": len(values) / denominator, "largeMisses": int((values > 5).sum()),
                "largeMissRate": float((values > 5).mean()), "maeFt": float(values.mean())}

    events = {event: summary(np.concatenate(parts), total[event]) for event, parts in errors.items()}
    result = summary(np.concatenate([array for parts in errors.values() for array in parts]), sum(total.values()))
    return {**result, "events": events}


def fingerprint():
    files = ("ARCHIVE_REVIEW_PROTOCOL.md", "archive_review.py", "map_model.py", "panel.csv")
    return {name: sha256((HERE / name).read_bytes()).hexdigest() for name in files}


def run(stage):
    lock = HERE / "archive_review_selection.json"
    panel = pd.read_csv(HERE / "panel.csv", keep_default_na=False)
    with threadpool_limits(4):
        model = fit_boosted(sample(panel, TRAIN, 18))
        if stage == "validation":
            dev = tuple(scores(panel[panel.event_id.isin(DEV)], model))
            by_budget = {}
            by_method = {}
            for fraction in FRACTIONS:
                reports = {method: summaries(dev, fraction, method) for method in METHODS}
                by_method[str(fraction)] = reports
                by_budget[str(fraction)] = min(METHODS, key=lambda method: reports[method]["largeMissRate"])
            result = {"freeze": fingerprint(), "primary": by_budget["0.2"],
                      "devBestByBudget": by_budget, "development": by_method}
            target = lock
        else:
            selected = json.loads(lock.read_text())
            if selected["freeze"] != fingerprint():
                raise ValueError("Review-policy lock has changed")
            archive_file = HERE / "archival_panel.csv"
            manifest = json.loads((HERE / "archival_manifest.json").read_text())
            if manifest["panelSha256"] != sha256(archive_file.read_bytes()).hexdigest():
                raise ValueError("Archival panel has changed")
            entries = tuple(scores(pd.read_csv(archive_file, keep_default_na=False), model))
            method = selected["primary"]
            result = {"policy": method, "curve": {str(p): summaries(entries, p, method) for p in FRACTIONS}}
            target = HERE / "archive_review_results.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "test"))
    run(parser.parse_args().stage)
