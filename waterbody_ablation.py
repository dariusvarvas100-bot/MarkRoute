"""Compare the released model with an otherwise identical fit without waterbody matches."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from map_model import SEED, TEST, TRAIN, fit_boosted, sample

HERE = Path(__file__).resolve().parent
FULL = "complete"
ABLATED = "without waterbody match"


def paired_interval(rows, *, ablated=ABLATED):
    per_event = []
    rng = np.random.default_rng(SEED)
    for _, event in rows.groupby("event"):
        blocks = event.assign(gain=event[ablated] - event[FULL]).groupby("block").gain.agg(["sum", "size"])
        draw = rng.integers(len(blocks), size=(4000, len(blocks)))
        per_event.append((blocks["sum"].to_numpy()[draw].sum(axis=1),
                          blocks["size"].to_numpy()[draw].sum(axis=1)))
    storms = rng.integers(len(per_event), size=(4000, len(per_event)))
    sums, counts = np.asarray(per_event).transpose(1, 2, 0)
    repeat = np.arange(4000)[:, None]
    changes = sums[repeat, storms].sum(axis=1) / counts[repeat, storms].sum(axis=1)
    return np.quantile(changes, [.025, .975]).tolist()


def summary(rows):
    full, ablated = (float(rows[name].mean()) for name in (FULL, ABLATED))
    gain = ablated - full
    interval = paired_interval(rows)
    return {
        "uniqueSites": int(rows.site.nunique()), "maskSitePairs": len(rows),
        "floodCount": int(rows.event.nunique()),
        "completeMaeFt": full, "withoutWaterbodyMatchMaeFt": ablated,
        "gainFt": gain, "relativeGain": gain / ablated, "paired95IntervalFt": interval,
        "eventWins": int((rows.groupby("event")[FULL].mean() < rows.groupby("event")[ABLATED].mean()).sum()),
        "meetsBenefitThreshold": bool(gain / ablated >= .02 and interval[0] > 0),
    }


def without_tags(frame):
    return frame.assign(waterbody="")


def run():
    panel = pd.read_csv(HERE / "panel.csv", keep_default_na=False)
    groups = [(panel, TEST)]
    for file in ("external_panel.csv", "remote_panel.csv"):
        extra = pd.read_csv(HERE / file, keep_default_na=False)
        groups.append((extra, list(extra.event_id.unique())))
    rows = []
    with threadpool_limits(4):
        full = fit_boosted(sample(panel, TRAIN, 18))
        ablated = fit_boosted(sample(without_tags(panel), TRAIN, 18))
        for frame, events in groups:
            a, b = sample(frame, events, 16), sample(without_tags(frame), events, 16)
            for key in ("actual", "site", "event", "environment", "block", "default", "baseline"):
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
            a_without_match = np.delete(a["features"], [3, 14, 25], axis=1)
            b_without_match = np.delete(b["features"], [3, 14, 25], axis=1)
            np.testing.assert_array_equal(a_without_match, b_without_match)
            np.testing.assert_array_equal(b["features"][:, [3, 14, 25]], 0)
            group = pd.DataFrame({key: a[key] for key in ("site", "event", "environment", "block")})
            for label, model, features in ((FULL, full, a), (ABLATED, ablated, b)):
                group[label] = np.abs(features["actual"] - features["default"] - model.predict(features["features"]))
            rows.append(group)
    combined = pd.concat(rows, ignore_index=True)
    output = {
        "freeze": {file: sha256((HERE / file).read_bytes()).hexdigest() for file in
                   ("WATERBODY_PROTOCOL.md", "waterbody_ablation.py", "map_model.py", "panel.csv",
                    "external_panel.csv", "remote_panel.csv")},
        "pooled": summary(combined),
        "environments": {label: summary(rows) for label, rows in combined.groupby("environment")},
        "eventMaeFt": combined.groupby("event")[[FULL, ABLATED]].mean().to_dict(orient="index"),
    }
    (HERE / "waterbody_ablation_results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run()
