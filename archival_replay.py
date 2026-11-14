"""Screen archived inland marks, freeze eligibility, and replay the deployed MarkRoute model."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
from subprocess import CalledProcessError

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from external_eval import SOURCE, get, terrain_tile
from kriging import BASELINE, ESTIMATOR, KRIGING, collect
from map_model import STATIC_NAMES, TRAIN, fit_boosted, sample
from terrain import web_coordinate
from waterbody_ablation import FULL, paired_interval

ROOT = Path(__file__).resolve().parent
ORIGINAL_PANELS = ("panel.csv", "external_panel.csv", "remote_panel.csv")
ARCHIVES = "earlier archives"
OVERLAP = "previously outside distance screen"
IOWA_GROUPS = (
    (143, 148),
    (145, 238, 276, 277, 278),
)
MINIMUM_SITES = 30
RELIEF_FT = 35
FIELDS = ["event_id", "event_name", "site_id", "latitude", "longitude", "elev_ft", "waterbody", "environment"]
EXCLUDE = re.compile(r"\b(test|training|shakedown|bridge|gages|study|exercise)\b", re.I)


def digest(file):
    return sha256((ROOT / file).read_bytes()).hexdigest()


def dump(file, value):
    (ROOT / file).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prior_data():
    return pd.concat([pd.read_csv(ROOT / name, keep_default_na=False) for name in ORIGINAL_PANELS])


def list_candidates(prior):
    records = json.loads((ROOT / "archival_catalog.json").read_text())
    prior_screen = json.loads((ROOT / "remote_manifest.json").read_text())
    overlapping = {row["id"] for row in prior_screen["eligibility"]
                   if row.get("excluded") == "fewer than 75% of sites beyond 100 km"}
    reserved = set(prior.event_id)
    candidates = []
    for record in records:
        event = record["id"]
        if event in reserved or not record["start_date"] or not record["end_date"]:
            continue
        start, end = record["start_date"][:10], record["end_date"][:10]
        historic = (record["event_type"] in (1, 2) and "2000-01-01" <= start <= end <= "2013-12-31" and
                    not EXCLUDE.search(record["name"]))
        if historic or event in overlapping:
            group = next((items[0] for items in IOWA_GROUPS if event in items), event)
            candidates.append({"id": event, "name": record["name"], "start": start, "end": end,
                               "group": group, "cohort": ARCHIVES if historic else OVERLAP,
                               "source": SOURCE.format(event=event)})
    return sorted(candidates, key=lambda row: row["id"])


def get_archive(event):
    path = ROOT / "source_marks" / f'{event["id"]}.json'
    try:
        raw = path.read_bytes() if path.exists() else get(event["source"])
        path.write_bytes(raw)
        decoded = json.loads(raw)
        event = {**event, "rawCount": len(decoded), "rawSha256": sha256(raw).hexdigest()}
        if not decoded:
            return {**event, "exclusion": "no archived marks"}, None
        points = pd.DataFrame(decoded)
        numeric = ["site_id", "latitude", "longitude", "elev_ft"]
        points[numeric] = points[numeric].apply(pd.to_numeric, errors="coerce")
        usable = (points.vertical_datum_name.eq("NAVD88") & points.hwm_quality_id.isin([1, 2, 3]) &
                  points.environment.eq("Riverine") & np.isfinite(points[numeric]).all(axis=1) &
                  points.latitude.between(-85, 85) & points.longitude.between(-180, 180))
        frame = points.loc[usable, ["id", "hwm_quality_id", *FIELDS[1:]]].copy()
        frame.insert(0, "event_id", event["group"])
        return {**event, "riverineRecords": len(frame)}, frame
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return {**event, "exclusion": f"archive error {type(error).__name__}"}, None
    except CalledProcessError:
        return {**event, "exclusion": "archive retrieval failed"}, None


def xyz(frame):
    lat, lon = np.deg2rad(frame[["latitude", "longitude"]].to_numpy()).T
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def screen(prior, sources):
    sets = {}
    for event, frame in sources:
        sets.setdefault(event["group"], []).append((event, frame))
    tree = cKDTree(xyz(prior))
    reports, eligible = [], []
    for event_id, source_group in sets.items():
        metadata = [info for info, _ in source_group]
        available = [frame for _, frame in source_group if frame is not None]
        frame = pd.concat(available, ignore_index=True) if available else pd.DataFrame(columns=FIELDS)
        label = next(info["name"] for info in metadata if info["id"] == event_id)
        frame = frame.loc[~frame.site_id.isin(prior.site_id)].copy()
        if len(frame):
            frame = frame.sort_values(["hwm_quality_id", "id"]).drop_duplicates("site_id")
            cells = np.floor(frame[["latitude", "longitude"]].to_numpy() * 600).astype(int)
            frame = frame.assign(loncell=cells[:, 1], latcell=cells[:, 0]).drop_duplicates(["loncell", "latcell"])
        report = {"id": event_id, "name": label, "cohort": metadata[0]["cohort"],
                  "sourceRecords": metadata, "distinctSites": len(frame)}
        if len(frame) < MINIMUM_SITES:
            report["excluded"] = "fewer than 30 independent riverine sites"
        else:
            frame = frame[FIELDS].assign(event_name=label, cohort=metadata[0]["cohort"])
            chord = tree.query(xyz(frame))[0]
            frame["distance_to_prior_km"] = 2 * 6371.0088 * np.arcsin(np.clip(chord / 2, 0, 1))
            eligible.append(frame)
        reports.append(report)
    if not eligible:
        raise ValueError("No eligible archive events")
    return pd.concat(eligible, ignore_index=True), reports


def add_terrain(points):
    (ROOT / "terrain_cache").mkdir(exist_ok=True)
    x, y = web_coordinate(points.latitude.to_numpy(), points.longitude.to_numpy())
    tiles = list(zip(x.astype(int) // 256, y.astype(int) // 256))
    with ThreadPoolExecutor(max_workers=8) as workers:
        rasters = {key: (raster, checksum) for key, raster, checksum in workers.map(terrain_tile, sorted(set(tiles)))}
    fields = []
    for tile, px, py in zip(tiles, x, y):
        raster, _ = rasters[tile]
        if raster is None:
            fields.append((np.nan, np.nan, np.nan))
            continue
        ix, iy = int(px) % 256, int(py) % 256
        local = raster[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]
        nearby = raster[max(0, iy - 8):iy + 9, max(0, ix - 8):ix + 9]
        fields.append((float(raster[iy, ix]), float(np.median(local)),
                       float(np.percentile(nearby, 90) - np.percentile(nearby, 10))))
    points[["terrain_ft", "terrain_median_ft", "relief_ft"]] = fields
    return points, {f"{key[0]}/{key[1]}": checksum for key, (_, checksum) in rasters.items()}


def prepare():
    prior = prior_data()
    candidates = list_candidates(prior)
    (ROOT / "source_marks").mkdir(exist_ok=True)
    dump("archival_eligibility_lock.json", {"protocolSha256": digest("ARCHIVAL_PROTOCOL.md"),
         "catalogSha256": digest("archival_catalog.json"), "candidates": candidates})
    print(f"Screening {len(candidates)} catalog records", flush=True)
    with ThreadPoolExecutor(max_workers=8) as workers:
        sources = list(workers.map(get_archive, candidates))
    points, reports = screen(prior, sources)
    print("Eligible before terrain:", [(r["name"], r["distinctSites"]) for r in reports if "excluded" not in r],
          flush=True)
    points, tiles = add_terrain(points)
    missing = set(points.loc[points.terrain_ft.isna(), "event_id"])
    for event in reports:
        if event["id"] in missing:
            event["excluded"] = "one or more terrain tiles unavailable"
    points = points.loc[~points.event_id.isin(missing)]
    points.to_csv(ROOT / "archival_panel.csv", index=False)
    dump("archival_manifest.json", {
        "protocolSha256": digest("ARCHIVAL_PROTOCOL.md"), "panelSha256": digest("archival_panel.csv"),
        "events": reports, "terrainTiles": tiles,
    })
    print(f"{len(points)} sites from {points.event_id.nunique()} retained events", flush=True)


def stats(frame):
    return {method: {"maeFt": float(frame[method].mean()), "overFiveFtRate": float(frame[method].gt(5).mean())}
            for method in (ESTIMATOR, BASELINE, KRIGING)}


def describe(frame):
    comparisons = {}
    summary = stats(frame)
    for method in (BASELINE, KRIGING):
        relative = 1 - summary[ESTIMATOR]["maeFt"] / summary[method]["maeFt"]
        interval = paired_interval(frame.rename(columns={ESTIMATOR: FULL}), ablated=method)
        comparisons[method] = {"relativeMaeReduction": relative, "paired95IntervalFt": interval}
    return {
        "events": int(frame.event.nunique()), "uniqueEventSites": int(len(frame[["event", "site"]].drop_duplicates())),
        "maskSitePairs": len(frame), "methods": summary, "comparisons": comparisons,
        "byEvent": {event: {"sites": int(group.site.nunique()), "maskSitePairs": len(group), "methods": stats(group)}
                    for event, group in frame.groupby("event")},
    }


def evaluate():
    manifest = json.loads((ROOT / "archival_manifest.json").read_text())
    if manifest["panelSha256"] != digest("archival_panel.csv") or manifest["protocolSha256"] != digest("ARCHIVAL_PROTOCOL.md"):
        raise ValueError("Archive panel or protocol has changed since eligibility screening")
    lock = json.loads((ROOT / "kriging_selection.json").read_text())
    if lock["originalPanelSha256"] != digest("panel.csv") or lock["scriptSha256"] != digest("kriging.py"):
        raise ValueError("Kriging selection has changed")
    if lock["modelFileSha256"] != digest("map_model.py"):
        raise ValueError("The frozen point model has changed")
    training = pd.read_csv(ROOT / "panel.csv", keep_default_na=False)
    points = pd.read_csv(ROOT / "archival_panel.csv", keep_default_na=False)
    assert set(points.event_id).isdisjoint(prior_data().event_id)
    assert set(points.site_id).isdisjoint(prior_data().site_id)
    with threadpool_limits(4):
        model = fit_boosted(sample(training, TRAIN, 18))
        events = points.event_id.unique()
        data = sample(points, events, 16)
        kriging = tuple(lock["selected"])
        predictors = {
            ESTIMATOR: model.predict(data["features"]) + data["default"],
            BASELINE: data["baseline"][:, STATIC_NAMES.index(BASELINE)],
            KRIGING: collect(points, events, 16, (kriging,), check=True)[kriging],
        }
    frame = pd.DataFrame({key: data[key] for key in ("event", "site", "block")})
    for name, values in predictors.items():
        frame[name] = np.abs(values - data["actual"])
    frame = frame.merge(points.rename(columns={"event_name": "event", "site_id": "site"})
                        [["event", "site", "cohort", "relief_ft", "distance_to_prior_km"]],
                        how="left", on=["event", "site"], validate="many_to_one")
    high_relief = frame[frame.relief_ft.ge(RELIEF_FT)]
    overall = describe(frame)
    group = describe(high_relief) if len(high_relief) else None
    passes = bool(group and group["events"] >= 2 and group["uniqueEventSites"] >= 30 and
                  all(value["relativeMaeReduction"] >= 0.05 and value["paired95IntervalFt"][0] > 0
                      for value in group["comparisons"].values()))
    dump("archival_results.json", {
        "manifestSha256": digest("archival_manifest.json"), "modelSha256": digest("map_model.py"),
        "highReliefPrimary": group, "passesHighReliefGate": passes, "all": overall,
        "byCohort": {cohort: describe(part) for cohort, part in frame.groupby("cohort")},
    })


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    {"prepare": prepare, "evaluate": evaluate}[parser.parse_args().stage]()
