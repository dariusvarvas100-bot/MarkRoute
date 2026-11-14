"""Replay the frozen MarkRoute policies on catalog-selected external events."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE

from map_model import SEED, STATIC_NAMES, bootstrap, fit_boosted, metrics, sample
from terrain import DEV, TEST, TRAIN, FEET_PER_METRE, ZOOM, web_coordinate


SOURCE = 'https://api.waterdata.usgs.gov/stnservices/hwms/filteredhwms/?event={event}&format=json'
TILE_URL = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png'
MATURITY = date(2026, 11, 14) - timedelta(days=120)
ORIGINAL_EVENTS = set((*TRAIN, *DEV, *TEST))
BASELINE = 'bounded local regression'


def candidate_catalog():
    cache = ORIGINAL / 'events_new.json'
    if not cache.is_file():
        cache.write_bytes(get('https://api.waterdata.usgs.gov/stnservices/events?format=json'))
    records = json.loads(cache.read_text())
    matched = []
    for record in records:
        identifier = record['id']
        if identifier in ORIGINAL_EVENTS or not record['start_date'] or not record['end_date']:
            continue
        started = date.fromisoformat(record['start_date'][:10])
        finished = date.fromisoformat(record['end_date'][:10])
        tropical = record['event_type'] == 2 and started >= date(2020, 1, 1)
        flood = record['event_type'] == 1 and started > date(2024, 9, 24)
        if (tropical or flood) and finished <= MATURITY:
            matched.append({'id': identifier, 'type': record['event_type'], 'event': record['name'],
                            'start': started.isoformat(), 'end': finished.isoformat()})
    return sorted(matched, key=lambda event: (event['start'], event['id']))


def get(url):
    return subprocess.check_output(['curl', '-fLsS', '--retry', '2', '--max-time', '25', url], stderr=subprocess.PIPE)


def get_marks(event):
    identifier = event['id']
    url = SOURCE.format(event=identifier)
    local = HERE / f'raw_{identifier}.json'
    earlier = ORIGINAL / f'event{identifier}.json'
    try:
        if local.is_file():
            content = local.read_bytes()
        elif earlier.is_file():
            content = earlier.read_bytes()
        else:
            content = get(url)
        decoded = json.loads(content)
        local.write_bytes(content)
        manifest = {**event, 'url': url, 'rawCount': len(decoded), 'rawSha256': sha256(content).hexdigest()}
        if not decoded:
            return {**manifest, 'excluded': 'no archived mark rows', 'eligibleCount': 0}, None
        data = pd.DataFrame(decoded)
        columns = ['event_id', 'event_name', 'site_id', 'latitude', 'longitude', 'elev_ft', 'waterbody', 'environment']
        selected = data[data.vertical_datum_name.eq('NAVD88') & data.hwm_quality_id.isin([1, 2, 3]) &
                        data.environment.isin(['Coastal', 'Riverine']) &
                        data[['latitude', 'longitude', 'elev_ft', 'site_id']].notna().all(axis=1)].copy()
        selected = selected.sort_values(['hwm_quality_id', 'id']).drop_duplicates('site_id')
        selected['loncell'] = np.floor(selected.longitude * 600).astype(int)
        selected['latcell'] = np.floor(selected.latitude * 600).astype(int)
        selected = selected.drop_duplicates(['loncell', 'latcell'])[columns].copy()
        selected['waterbody'] = selected.waterbody.fillna('').str.casefold().str.strip().str.replace(r'[^\w]+', ' ', regex=True)
        selected.loc[selected.waterbody.isin(['unknown', 'n a', 'na', 'unnamed']), 'waterbody'] = ''
        manifest['eligibleCount'] = len(selected)
        if len(selected) < 30:
            return {**manifest, 'excluded': 'less than 30 eligible marks'}, None
        return manifest, selected
    except (subprocess.CalledProcessError, UnicodeError, OSError, ValueError, KeyError, AttributeError) as exc:
        return {**event, 'url': url, 'excluded': 'source unavailable or malformed', 'errorType': type(exc).__name__}, None


def terrain_tile(key):
    x, y = key
    file = f'{x}_{y}.png'
    local = HERE / 'terrain_cache' / file
    existing = ORIGINAL / 'terrain_cache' / file
    url = TILE_URL.format(zoom=ZOOM, x=x, y=y)
    try:
        if local.is_file():
            content = local.read_bytes()
        elif existing.is_file():
            content = existing.read_bytes()
        else:
            content = get(url)
        local.write_bytes(content)
        with Image.open(local) as image:
            red, green, blue = np.asarray(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1)
        return key, (red * 256 + green + blue / 256 - 32768) * FEET_PER_METRE, sha256(content).hexdigest()
    except (subprocess.CalledProcessError, OSError, ValueError):
        return key, None, None


def prepare():
    events = candidate_catalog()
    print('Frozen event IDs:', [event['id'] for event in events], flush=True)
    with ThreadPoolExecutor(max_workers=4) as workers:
        sources = list(workers.map(get_marks, events))
    print([(report['id'], report.get('eligibleCount'), report.get('excluded')) for report, _ in sources], flush=True)
    eligible = [frame for _, frame in sources if frame is not None]
    frame = pd.concat(eligible, ignore_index=True)
    cx, cy = web_coordinate(frame.latitude.to_numpy(), frame.longitude.to_numpy())
    keys = list(zip(cx.astype(int)//256, cy.astype(int)//256))
    print(len(frame), 'retained marks and', len(set(keys)), 'terrain tile keys', flush=True)
    (HERE / 'terrain_cache').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as workers:
        rasters = {key: (raster, digest) for key, raster, digest in workers.map(terrain_tile, sorted(set(keys)))}
    features = []
    for key, x, y in zip(keys, cx, cy):
        raster, _ = rasters[key]
        if raster is None:
            features.append((np.nan, np.nan, np.nan))
            continue
        ix, iy = int(x) % 256, int(y) % 256
        local = raster[max(0, iy-3):iy+4, max(0, ix-3):ix+4]
        landscape = raster[max(0, iy-8):iy+9, max(0, ix-8):ix+9]
        features.append((raster[iy, ix], float(np.median(local)), float(np.percentile(landscape, 90)-np.percentile(landscape, 10))))
    frame[['terrain_ft', 'terrain_median_ft', 'relief_ft']] = features
    missing_terrain = set(frame.loc[frame.terrain_ft.isna(), 'event_id'])
    frame = frame.loc[~frame.event_id.isin(missing_terrain)]
    for report, _ in sources:
        if report['id'] in missing_terrain:
            report['excluded'] = 'one or more source tiles unavailable'
    frame.to_csv(HERE / 'external_panel.csv', index=False)
    audit = {'maturityCutoff': MATURITY.isoformat(), 'eventCatalogSha256': sha256((ORIGINAL/'events_new.json').read_bytes()).hexdigest(),
             'events': [report for report, _ in sources], 'terrainSha256': {f'{x}/{y}': digest for (x, y), (_, digest) in rasters.items()}}
    (HERE / 'external_manifest.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('External panel rows:', len(frame), flush=True)


def fit_boost():
    original = pd.read_csv(ORIGINAL / 'panel.csv', keep_default_na=False)
    return fit_boosted(sample(original, TRAIN, 18))


def evaluate():
    import torch
    torch.set_num_threads(4)
    model = fit_boost()
    results = {}
    for title, path, ids in [('originalIndependentReplay', ORIGINAL / 'panel.csv', TEST),
                             ('additionalEventReplay', HERE / 'external_panel.csv', None)]:
        frame = pd.read_csv(path, keep_default_na=False)
        cohort = ids or tuple(frame.event_id.unique())
        held = sample(frame, cohort, 16)
        baseline = held['baseline'][:, STATIC_NAMES.index(BASELINE)]
        boosted = model.predict(held['features']) + held['default']
        summary = metrics({BASELINE: baseline, 'gradient boost': boosted}, held)
        base_abs, model_abs = np.abs(held['actual'] - baseline), np.abs(held['actual'] - boosted)
        event_names = np.unique(held['event'])
        errors = {event: (base_abs[held['event']==event].mean(), model_abs[held['event']==event].mean()) for event in event_names}
        relative = {event: 1 - candidate / control for event, (control, candidate) in errors.items()}
        pairs = np.asarray(list(errors.values()))
        rng = np.random.default_rng(SEED + 230)
        sampled = pairs[rng.integers(len(pairs), size=(4000, len(pairs)))].mean(axis=1)
        result = {'distinctSites': int(len(np.unique(held['site']))), 'maskSitePairs': len(held['site']),
                  'events': summary['events'], 'environments': summary['environments'], 'pooled': summary['pooled'],
                  'eventPairCounts': {event: int(np.count_nonzero(held['event']==event)) for event in event_names},
                  'testEvents': int(len(cohort)), 'relativeGainByEvent': relative,
                  'pooledRelativeGain': float(1 - model_abs.mean()/base_abs.mean()),
                  'macroRelativeGain': float(1 - pairs[:,1].mean()/pairs[:,0].mean()),
                  'eventResampledMacroAbsolute95Ft': np.quantile(sampled[:,0]-sampled[:,1], [.025, .975]).tolist(),
                  'spatialPairedAbsolute95Ft': bootstrap(held, boosted, baseline),
                  'pooledAbsoluteError90and99Ft': {BASELINE: np.quantile(base_abs,[.9,.99]).tolist(),
                                                   'gradient boost': np.quantile(model_abs,[.9,.99]).tolist()}}
        result['passed'] = bool(result['pooledRelativeGain'] >= .10 and min(relative.values()) >= -.05 and result['spatialPairedAbsolute95Ft'][0] > 0)
        results[title] = result
    (HERE / 'external_results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('catalog', 'prepare', 'evaluate'))
    mode = parser.parse_args().stage
    if mode == 'catalog':
        print(json.dumps(candidate_catalog(), indent=2))
    elif mode == 'prepare':
        prepare()
    else:
        evaluate()
