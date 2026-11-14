"""Replay the frozen mapper with nested survey budgets and spatial gaps."""

from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / 'map_model.py').is_file() else HERE.parent
sys.path.insert(0, str(ROOT))

from map_model import DEV, SEED, STATIC_NAMES, TEST, TRAIN, bootstrap, fit_boosted, observation_features, sample, split_points, xy_km


BUDGETS = (5, 10, 15, 25, 40)
REPLAYS = 16
BASELINE = 'bounded local regression'
MODEL = 'gradient boost'
INITIAL_SEED = 99173
GAP_SEED = 313191


def quota(population, percent):
    return min(population - 1, max(2, round(population * percent / 100)))


def nested_partitions(frame, env, seed):
    original_anchors, _ = split_points(frame, seed)
    rng = np.random.default_rng(INITIAL_SEED + seed)
    ordering = {}
    for kind, ids in env.items():
        original = np.intersect1d(original_anchors, ids)
        remainder = np.setdiff1d(ids, original)
        ordering[kind] = np.r_[rng.permutation(original), rng.permutation(remainder)]
        assert len(original) == quota(len(ids), 15)
    anchors = {percent: np.sort(np.concatenate([ids[:quota(len(ids), percent)] for ids in ordering.values()]))
               for percent in BUDGETS}
    assert np.array_equal(anchors[15], original_anchors)
    assert all(np.isin(anchors[a], anchors[b]).all() for a, b in zip(BUDGETS, BUDGETS[1:]))
    return anchors


def gap_partitions(frame, env, distances, seed):
    rng = np.random.default_rng(GAP_SEED + seed)
    n_sites = len(frame)
    n_patch = round(n_sites * .2)
    all_sites = np.arange(n_sites)
    for center in rng.permutation(n_sites):
        patch = np.argsort(distances[center], kind='stable')[:n_patch]
        available = ~np.isin(all_sites, patch)
        if all(np.count_nonzero(available[ids]) >= quota(len(ids), 15) for ids in env.values()):
            break
    else:
        raise ValueError('Could not withhold a patch and preserve the survey allowance')
    scoring = np.sort(rng.choice(patch, n_patch // 2, replace=False))
    dispersed = []
    clustered = []
    for ids in env.values():
        count = quota(len(ids), 15)
        order = rng.permutation(ids)
        dispersed.extend(order[~np.isin(order, scoring)][:count])
        clustered.extend(order[available[order]][:count])
    dispersed = np.sort(dispersed)
    clustered = np.sort(clustered)
    assert len(dispersed) == len(clustered)
    assert not np.intersect1d(clustered, patch).size
    assert not np.intersect1d(dispersed, scoring).size
    return dispersed, clustered, scoring, patch


def prepare_features(frame, environment, coords, distances, anchors, targets):
    if np.intersect1d(anchors, targets).size or not len(anchors):
        raise ValueError('Invalid reveal/score partition')
    result = []
    features = []
    defaults = []
    for label, ids in environment.items():
        requested = np.intersect1d(ids, targets)
        if not len(requested):
            continue
        context = np.intersect1d(anchors, ids)
        if not len(context):
            context = anchors
        data = observation_features(frame, coords, distances, context, requested)
        outcome = data['actual']
        base = data['baseline'][:, STATIC_NAMES.index(BASELINE)]
        result.append(pd.DataFrame({'target_position': requested, 'environment': label,
                                    'outcome': outcome, BASELINE: base}))
        features.append(data['features'])
        defaults.append(data['default'])
    rows = pd.concat(result, ignore_index=True)
    ordering = np.argsort(rows.target_position.to_numpy())
    rows = rows.iloc[ordering]
    assert np.array_equal(rows.target_position.to_numpy(), targets)
    return rows, np.concatenate(features)[ordering], np.concatenate(defaults)[ordering]


def annotate(predictions, event, seed, coords):
    positions = predictions.target_position.to_numpy()
    blocks = [f'{int(x//10)}_{int(y//10)}' for x, y in coords[positions]]
    return predictions.assign(event=event, seed=seed, block=blocks)


def collect(model, cohorts):
    natural = {n: [] for n in BUDGETS}
    common = {n: [] for n in BUDGETS}
    gaps = {'dispersed': [], 'missing_cluster': []}
    partition_audit = {}
    for event, frame in cohorts:
        started = perf_counter()
        xy = xy_km(frame)
        distances = cdist(xy, xy)
        environment = frame.groupby('environment', sort=True).indices
        all_sites = np.arange(len(frame))
        gap_audit = []
        patch_anchor_counts = []
        anchor_distances = [[], []]
        pending = []
        for seed in range(SEED, SEED + REPLAYS):
            anchors = nested_partitions(frame, environment, seed)
            fixed_targets = np.setdiff1d(all_sites, anchors[40])
            for percent, seen in anchors.items():
                hidden = np.setdiff1d(all_sites, seen)
                rows, data, center = prepare_features(frame, environment, xy, distances, seen, hidden)
                pending.append((percent, annotate(rows, event, seed, xy), data, center, fixed_targets))
            dispersed, missing, scoring, patch = gap_partitions(frame, environment, distances, seed)
            for scenario, (label, seen) in enumerate(zip(gaps, (dispersed, missing))):
                rows, data, center = prepare_features(frame, environment, xy, distances, seen, scoring)
                pending.append((label, annotate(rows, event, seed, xy), data, center, None))
                anchor_distances[scenario].extend(distances[np.ix_(scoring, seen)].min(axis=1).tolist())
            gap_audit.append((len(patch), len(scoring), len(dispersed)))
            patch_anchor_counts.append(int(np.intersect1d(patch, dispersed).size))
            if seed == SEED:
                hidden = np.setdiff1d(all_sites, missing)
                changed = frame.copy()
                changed.loc[hidden, 'elev_ft'] = 1e8 + hidden.astype(float)
                expected, x, shift = prepare_features(frame, environment, xy, distances, missing, scoring)
                probe, compare_x, compare_shift = prepare_features(changed, environment, xy, distances, missing, scoring)
                assert expected[[BASELINE]].equals(probe[[BASELINE]])
                assert np.array_equal(x, compare_x) and np.array_equal(shift, compare_shift)
                assert (expected.outcome != probe.outcome).all()
        print(f'{event}: prepared {len(pending)} matched batches', flush=True)
        estimates = model.predict(np.concatenate([data for _, _, data, _, _ in pending]))
        offset = 0
        for label, rows, _, default, fixed in pending:
            end = offset + len(rows)
            rows[MODEL] = estimates[offset:end] + default
            if fixed is None:
                gaps[label].append(rows)
            else:
                natural[label].append(rows)
                common[label].append(rows.loc[rows.target_position.isin(fixed)])
            offset = end
        assert len(set(gap_audit)) == 1
        patch_size, scored, gap_anchors = gap_audit[0]
        partition_audit[event] = {'sites': len(frame), 'anchorsByNominalPercent': {n: len(a) for n, a in anchors.items()},
                                  'commonTargetsPerMask': len(fixed_targets), 'gapPatchPerMask': patch_size,
                                  'gapScoredPerMask': scored, 'gapAnchorsPerMask': gap_anchors,
                                  'dispersedAnchorsInPatchMean': float(np.mean(patch_anchor_counts)),
                                  'nearestAnchorMedianKm': dict(zip(gaps, [float(np.median(d)) for d in anchor_distances]))}
        print(f'{event}: {len(frame)} sites, {REPLAYS} masks, {perf_counter() - started:.2f} seconds', flush=True)
    combine = lambda table: {name: pd.concat(parts, ignore_index=True) for name, parts in table.items()}
    return combine(natural), combine(common), combine(gaps), partition_audit


def pair_interval(data, a, b):
    inputs = {'actual': data.outcome.to_numpy(), 'event': data.event.to_numpy(), 'block': data.block.to_numpy()}
    return bootstrap(inputs, data[a].to_numpy(), data[b].to_numpy())


def metrics(rows):
    details = {}
    for method in (BASELINE, MODEL):
        errors = np.abs(rows[method].to_numpy() - rows.outcome.to_numpy())
        details[method] = {'maeFt': float(errors.mean()), 'over5Ft': int(np.count_nonzero(errors > 5)),
                           'over5FtRate': float(np.mean(errors > 5)), 'p90Ft': float(np.quantile(errors,.90)),
                           'p99Ft': float(np.quantile(errors,.99))}
    return {'maskSitePairs': len(rows), **details,
            'relativeMaeReduction': 1 - details[MODEL]['maeFt'] / details[BASELINE]['maeFt']}


def summarize(rows):
    return {'pooled': metrics(rows),
            'conditionalPairedSpatial95GainFt': pair_interval(rows, MODEL, BASELINE),
            'events': {event: metrics(group) for event, group in rows.groupby('event', sort=True)},
            'environments': {label: metrics(group) for label, group in rows.groupby('environment', sort=True)}}


def audit_original(rows):
    original = json.loads((ROOT / 'holdout.json').read_text())['results']['events']
    extension = json.loads(packaged_or_work('external_results.json', ROOT / 'external' / 'evaluation.json').read_text())['additionalEventReplay']['events']
    frozen = original | extension
    actual = {event: metrics(group) for event, group in rows.groupby('event')}
    for event, methods in actual.items():
        for method in (BASELINE, MODEL):
            if not np.isclose(frozen[event][method]['maeFt'], methods[method]['maeFt'], atol=1e-10, rtol=0):
                raise AssertionError(f'Original 15% replay mismatch for {event}, {method}')
    return True


def gap_summary(gaps):
    dispersed = gaps['dispersed'].sort_values(['event', 'seed', 'target_position']).reset_index(drop=True)
    absent = gaps['missing_cluster'].sort_values(['event', 'seed', 'target_position']).reset_index(drop=True)
    assert dispersed[['event','seed','target_position','outcome']].equals(absent[['event','seed','target_position','outcome']])
    intervals = {}
    for name in (MODEL, BASELINE):
        comparison = dispersed.assign(gap=absent[name])
        intervals[name] = {'increaseMaeFt': metrics(absent)[name]['maeFt'] - metrics(dispersed)[name]['maeFt'],
                           'conditionalPairedSpatial95ChangeFt': pair_interval(comparison, name, 'gap')}
    return {'dispersed': summarize(dispersed), 'missingCluster': summarize(absent), 'missingVersusDispersed': intervals}


def file_hash(file):
    return sha256(file.read_bytes()).hexdigest()


def packaged_or_work(name, fallback):
    shipped = ROOT / name
    return shipped if shipped.is_file() else fallback


def main():
    total_start = perf_counter()
    torch.set_num_threads(4)
    original_path = ROOT / 'panel.csv'
    external_path = packaged_or_work('external_panel.csv', ROOT / 'external' / 'panel.csv')
    original = pd.read_csv(original_path, keep_default_na=False)
    external = pd.read_csv(external_path, keep_default_na=False)
    assert set(external.event_id).isdisjoint((*TRAIN, *DEV, *TEST))
    print('Refitting the original frozen gradient-boosted model', flush=True)
    fit_start = perf_counter()
    model = fit_boosted(sample(original, TRAIN, 18))
    fit_end = perf_counter()
    test = pd.concat([original.loc[original.event_id.isin(TEST)], external], ignore_index=True)
    cohorts = [(event, frame.reset_index(drop=True)) for event, frame in test.groupby('event_name', sort=True)]
    natural, common, gaps, partitions = collect(model, cohorts)
    scoring_done = perf_counter()
    payload = {'protocol': 'ROBUSTNESS_PROTOCOL.md', 'seedsPerEvent': REPLAYS, 'nominalBudgetsPercent': BUDGETS,
               'original15PercentPerEventVerified': audit_original(natural[15]),
               'gapHiddenOutcomeSentinelPassedPerEvent': True,
               'inputSha256': {'originalPanel': file_hash(original_path), 'externalPanel': file_hash(external_path),
                               'frozenMappingScript': file_hash(ROOT / 'map_model.py')},
               'stationsByEvent': partitions, 'allRemaining': {n: summarize(rows) for n, rows in natural.items()},
               'commonTargets': {n: summarize(rows) for n, rows in common.items()}, 'geographicGap': gap_summary(gaps)}
    results_name = 'robustness_results.json' if HERE == ROOT else 'results.json'
    timing_name = 'robustness_timing.json' if HERE == ROOT else 'timing.json'
    (HERE / results_name).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    elapsed = {'python': platform.python_version(), 'machine': platform.machine(), 'secondsRefit': fit_end-fit_start,
               'secondsScoring': scoring_done-fit_end, 'secondsTotal': perf_counter()-total_start}
    (HERE / timing_name).write_text(json.dumps(elapsed, indent=2) + '\n')
    for percent, row in payload['commonTargets'].items():
        print(f"{percent}% common target {row['pooled'][MODEL]['maeFt']:.3f} vs {row['pooled'][BASELINE]['maeFt']:.3f} ft")
    print(json.dumps(payload['geographicGap']['missingVersusDispersed'], indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
