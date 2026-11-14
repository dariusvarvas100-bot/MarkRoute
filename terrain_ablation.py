"""Measure the value of terrain on the geographically distant survey replays."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from map_model import TRAIN, fit_boosted, sample
from waterbody_ablation import FULL, paired_interval

HERE = Path(__file__).resolve().parent
WITHOUT = "without terrain"
FIELDS = ("terrain_ft", "terrain_median_ft", "relief_ft")


def without_terrain(frame):
    return frame.assign(**dict.fromkeys(FIELDS, 0.0))


def run():
    train = pd.read_csv(HERE / "panel.csv", keep_default_na=False)
    remote = pd.read_csv(HERE / "remote_panel.csv", keep_default_na=False)
    storms = remote.event_id.unique()
    with threadpool_limits(4):
        full_model = fit_boosted(sample(train, TRAIN, 18))
        ablated_model = fit_boosted(sample(without_terrain(train), TRAIN, 18))
        complete, ablated = sample(remote, storms, 16), sample(without_terrain(remote), storms, 16)
        for key in ("actual", "site", "event", "environment", "block"):
            np.testing.assert_array_equal(complete[key], ablated[key], err_msg=key)
        errors = pd.DataFrame({key: complete[key] for key in ("site", "event", "environment", "block")})
        errors[FULL] = np.abs(complete["actual"] - complete["default"] - full_model.predict(complete["features"]))
        errors[WITHOUT] = np.abs(ablated["actual"] - ablated["default"] - ablated_model.predict(ablated["features"]))
    primary = errors[errors.environment.eq("Riverine")]
    full_mae, ablated_mae = (float(primary[key].mean()) for key in (FULL, WITHOUT))
    relative = (ablated_mae - full_mae) / ablated_mae
    interval = paired_interval(primary, ablated=WITHOUT)
    result = {
        "freeze": {file: sha256((HERE / file).read_bytes()).hexdigest() for file in
                   ("TERRAIN_ABLATION_PROTOCOL.md", "terrain_ablation.py", "map_model.py",
                    "waterbody_ablation.py", "panel.csv", "remote_panel.csv")},
        "primaryRiverine": {
            "sites": int(primary.site.nunique()), "maskSitePairs": len(primary),
            "completeMaeFt": full_mae, "withoutTerrainMaeFt": ablated_mae,
            "relativeGain": relative, "paired95IntervalFt": interval,
            "meetsBenefitThreshold": bool(relative >= .05 and interval[0] > 0),
            "events": primary.groupby("event")[[FULL, WITHOUT]].mean().to_dict(orient="index"),
        },
        "environments": errors.groupby("environment")[[FULL, WITHOUT]].mean().to_dict(orient="index"),
    }
    (HERE / "terrain_ablation_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run()
