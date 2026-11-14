"""Evaluate a development-selected kriging comparator on MarkRoute's frozen masks."""
from argparse import ArgumentParser
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


HERE = Path(__file__).resolve().parent
APP = HERE

from map_model import DEV, PEERS, SEED, STATIC_NAMES, TEST, TRAIN, bootstrap, fit_boosted, metrics, sample, split_points, xy_km  # noqa: E402


BASELINE = "bounded local regression"
ESTIMATOR = "gradient boost"
KRIGING = "terrain-aware local kriging"
SETTINGS = tuple(product(("physics", "bounded"), (.75, 2.5), (None, 45.0), (.03, .25)))


def source_hash(file):
    return sha256(file.read_bytes()).hexdigest()


def matern32(distance):
    scaled = np.sqrt(3) * distance
    return (1 + scaled) * np.exp(-scaled)


def anchors_for_cohort(frame, xy, distance, context, targets):
    nearest = np.argsort(distance[np.ix_(targets, context)], axis=1)[:, :PEERS]
    indices = context[nearest]
    separation = distance[targets[:, None], indices]
    site_ground = frame.terrain_median_ft.to_numpy()
    waterbody = frame.waterbody.to_numpy()
    measured = frame.elev_ft.to_numpy()[indices]
    heights = site_ground[indices]
    candidate_height = site_ground[targets]
    coordinates = xy[indices]
    sector = waterbody[indices]
    defined = sector != ""
    connected = (sector[:, :, None] == sector[:, None, :]) & defined[:, :, None] & defined[:, None, :]
    target_connected = (sector == waterbody[targets, None]) & defined
    p = 1 / (.25 + separation) ** 1.5
    p /= p.sum(axis=1, keepdims=True)
    centered_height = heights - (p * heights).sum(axis=1, keepdims=True)
    centered_elevation = measured - (p * measured).sum(axis=1, keepdims=True)
    local_slope = np.clip((p * centered_height * centered_elevation).sum(axis=1) /
                          ((p * centered_height ** 2).sum(axis=1) + 4), 0, 1.2)
    return {
        "targetGround": candidate_height, "anchorGround": heights, "measurements": measured,
        "horizontal": separation, "blockDistance": np.linalg.norm(coordinates[:, :, None] - coordinates[:, None, :], axis=-1),
        "localSlope": local_slope, "physicsSlope": frame.environment.to_numpy()[targets] != "Coastal",
        "blockWater": .35 + .65 * connected, "queryWater": .35 + .65 * target_connected,
    }


def estimate(context, setting):
    trend, horizontal_factor, vertical_scale, noise = setting
    ground = context["anchorGround"]
    query_ground = context["targetGround"]
    horizontal = context["horizontal"]
    length = np.maximum(.25, horizontal_factor * np.median(horizontal, axis=1))
    cross = horizontal / length[:, None]
    inside = context["blockDistance"] / length[:, None, None]
    if vertical_scale is not None:
        cross = np.hypot(cross, (ground - query_ground[:, None]) / vertical_scale)
        inside = np.hypot(inside, (ground[:, :, None] - ground[:, None, :]) / vertical_scale)
    kernel = matern32(inside) * context["blockWater"]
    diagonal = np.diag_indices(kernel.shape[1])
    kernel[:, diagonal[0], diagonal[1]] = 1 + noise
    cross_covariance = matern32(cross) * context["queryWater"]
    slope = context["physicsSlope"] if trend == "physics" else context["localSlope"]
    residual = context["measurements"] - slope[:, None] * ground
    right_hand = np.stack((residual, np.ones_like(residual)), axis=-1)
    solved = np.linalg.solve(kernel, right_hand)
    mean = solved[:, :, 0].sum(axis=1) / solved[:, :, 1].sum(axis=1)
    weights = solved[:, :, 0] - mean[:, None] * solved[:, :, 1]
    return slope * query_ground + mean + (cross_covariance * weights).sum(axis=1)


def collect(frame, storms, repetitions, settings, *, check=False):
    parts = {setting: [] for setting in settings}
    for storm in storms:
        points = frame[frame.event_id.eq(storm)].reset_index(drop=True)
        coordinates = xy_km(points)
        distances = cdist(coordinates, coordinates)
        groups = points.groupby("environment").indices
        for seed in range(SEED, SEED + repetitions):
            seen, queries = split_points(points, seed)
            for ids in groups.values():
                targets = np.intersect1d(queries, ids)
                if not len(targets):
                    continue
                anchors = np.intersect1d(seen, ids)
                if not len(anchors):
                    anchors = seen
                context = anchors_for_cohort(points, coordinates, distances, anchors, targets)
                test_hidden = check and seed == SEED
                if test_hidden:
                    blinded = points.copy()
                    blinded.loc[queries, "elev_ft"] = np.nan
                    hidden = anchors_for_cohort(blinded, coordinates, distances, anchors, targets)
                for setting in settings:
                    predicted = estimate(context, setting)
                    if not np.isfinite(predicted).all():
                        raise ValueError("A target has no finite kriging prediction.")
                    if test_hidden:
                        np.testing.assert_array_equal(estimate(hidden, setting), predicted)
                    parts[setting].append(predicted)
    return {setting: np.concatenate(predictions) for setting, predictions in parts.items()}


def synthetic_check():
    points = pd.DataFrame({
        "environment": ["Riverine"] * 4, "terrain_median_ft": [1., 5., 21., 42.],
        "elev_ft": [5., 9., 25., 46.], "waterbody": ["a", "a", "", ""],
    })
    xy = np.array([[0, 0], [1, 3], [7, 2], [4, 7]])
    anchors = np.array([0, 1, 2])
    targets = np.array([3])
    context = anchors_for_cohort(points, xy, cdist(xy, xy), anchors, targets)
    np.testing.assert_allclose(estimate(context, ("physics", .75, 45., .25)), [46.], atol=1e-10)
    points["elev_ft"] = 37.
    points["environment"] = "Coastal"
    context = anchors_for_cohort(points, xy, cdist(xy, xy), anchors, targets)
    np.testing.assert_allclose(estimate(context, ("physics", .75, 45., .25)), [37.], atol=1e-10)


def summarize(data, predicted):
    result = metrics(predicted, data)
    result["distinctSites"] = int(len(np.unique(data["site"])))
    result["predictivePairs"] = len(data["site"])
    result["macroAverageMaeFt"] = {
        method: float(np.mean([entry[method]["maeFt"] for entry in result["events"].values()]))
        for method in predicted
    }
    result["markRouteGainOverKriging95Ft"] = bootstrap(data, predicted[ESTIMATOR], predicted[KRIGING])
    result["krigingGainOverBounded95Ft"] = bootstrap(data, predicted[KRIGING], predicted[BASELINE])
    return result


def benchmark_one(booster, frame, storms, selected, expected):
    data = sample(frame, storms, 16)
    ranked = collect(frame, storms, 16, (selected,), check=True)[selected]
    bound = data["baseline"][:, STATIC_NAMES.index(BASELINE)]
    boosted = booster.predict(data["features"]) + data["default"]
    scored = {BASELINE: bound, ESTIMATOR: boosted, KRIGING: ranked}
    result = summarize(data, scored)
    assert result["distinctSites"] == expected["distinctSites"]
    assert result["predictivePairs"] == expected["maskSitePairs"]
    for event, scores in result["events"].items():
        for method in (BASELINE, ESTIMATOR):
            np.testing.assert_allclose(scores[method]["maeFt"], expected["events"][event][method]["maeFt"], atol=1e-9)
    return result, data, scored


def run(stage):
    synthetic_check()
    original_path = APP / "panel.csv"
    original = pd.read_csv(original_path, keep_default_na=False)
    settings_path = HERE / "kriging_selection.json"
    if stage == "validation":
        dev = sample(original, DEV, 10)
        candidate = collect(original, DEV, 10, SETTINGS, check=True)
        errors = {setting: float(np.abs(prediction - dev["actual"]).mean())
                  for setting, prediction in candidate.items()}
        best = min(errors, key=errors.get)
        result = {"selected": best, "selectionMaeFt": errors[best],
                  "choices": [{"setting": setting, "validationMaeFt": error} for setting, error in errors.items()],
                  "originalPanelSha256": source_hash(original_path), "scriptSha256": source_hash(Path(__file__)),
                  "modelFileSha256": source_hash(APP / "map_model.py")}
        output_path = settings_path
    else:
        lock = json.loads(settings_path.read_text())
        assert lock["originalPanelSha256"] == source_hash(original_path)
        assert lock["scriptSha256"] == source_hash(Path(__file__))
        assert lock["modelFileSha256"] == source_hash(APP / "map_model.py")
        selection = tuple(lock["selected"])
        assert selection in SETTINGS
        archived = json.loads((APP / "external_results.json").read_text())
        booster = fit_boosted(sample(original, TRAIN, 18))
        external = pd.read_csv(APP / "external_panel.csv", keep_default_na=False)
        groups = {"originalIndependentReplay": (original, TEST),
                  "additionalEventReplay": (external, tuple(external.event_id.unique()))}
        joined = []
        result = {"selectedSetting": selection}
        for name, (frame, storms) in groups.items():
            report, data, predictions = benchmark_one(booster, frame, storms, selection, archived[name])
            result[name] = report
            joined.append((data, predictions))
        keys = ("actual", "event", "environment", "site", "block")
        pooled_data = {key: np.concatenate([data[key] for data, _ in joined]) for key in keys}
        pooled_preds = {key: np.concatenate([prediction[key] for _, prediction in joined])
                        for key in (BASELINE, ESTIMATOR, KRIGING)}
        result["combinedReplay"] = summarize(pooled_data, pooled_preds)
        output_path = HERE / "kriging_results.json"
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output_path.write_text(text)
    print(text, end="", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validation", "test"))
    run(parser.parse_args().stage)
