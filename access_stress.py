"""Replay MarkRoute crew batches under deterministic synthetic candidate closures."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
OPEN = 'unrestricted'
SCATTERED = 'scattered 30%'
CLUSTERED = 'clustered 30%'
SCENARIOS = (OPEN, SCATTERED, CLUSTERED)
BASE = 'no extra'
POLICY = 'terrain coverage'
READING_PLANS = (POLICY, 'nearest', 'farthest', 'random')
COMMON = ['event', 'environment', 'site', 'mask', 'block', 'reliefFt']


def candidate_access(event_id, environment, seed, scenario, candidates, distance):
    if scenario == OPEN:
        return candidates, np.array([], dtype=int), None
    key = f'{event_id}|{environment}|{seed}|{scenario}'.encode()
    rng = np.random.default_rng(int.from_bytes(sha256(key).digest()[:8], 'little'))
    blocked_count = (3 * len(candidates) + 5) // 10
    if scenario == SCATTERED:
        blocked = rng.choice(candidates, blocked_count, replace=False)
        center = None
    else:
        center = int(rng.choice(candidates))
        blocked = candidates[np.argsort(distance[center, candidates], kind='stable')[:blocked_count]]
    eligible = candidates[~np.isin(candidates, blocked)]
    assert len(eligible) + blocked_count == len(candidates)
    return eligible, blocked, center


def make_plans(surface, distances, anchors, eligible, scoring, random_order, count):
    from survey_decision import greedy_coverage, kernel

    terrain = kernel(surface, distances, anchors, np.r_[eligible, scoring], .75)
    farthest, coverage = greedy_coverage(distances, terrain, anchors, eligible, scoring, count)
    distance = distances[np.ix_(eligible, anchors)].min(axis=1)
    nearest = eligible[np.argsort(distance, kind='stable')[:count]]
    random = random_order[np.isin(random_order, eligible)][:count]
    return {BASE: np.array([], dtype=int), POLICY: coverage, 'nearest': nearest, 'farthest': farthest, 'random': random}


def replay_event(frame, model, fixed_open, case_seed=None):
    from crew_replay import stats
    from map_model import observation_features, xy_km
    from survey_decision import MASKS, partition

    frame = frame.reset_index(drop=True)
    xy = xy_km(frame)
    distances = cdist(xy, xy)
    elevations = frame.elev_ft.to_numpy()
    site = frame.site_id.to_numpy()
    environments = frame.environment.to_numpy()
    regions = frame.relief_ft.to_numpy()
    event_id, event_name = frame.event_id.iat[0], frame.event_name.iat[0]
    rows = {scenario: [] for scenario in SCENARIOS}
    budgets = {scenario: [] for scenario in SCENARIOS}
    case = {scenario: [] for scenario in SCENARIOS}
    for seed in MASKS:
        for anchors, candidates, scoring, count in partition(frame, seed):
            environment = environments[scoring[0]]
            assert len(set(anchors) | set(candidates) | set(scoring)) == len(anchors) + len(candidates) + len(scoring)
            surface = frame.assign(elev_ft=np.nan)
            surface.loc[anchors, 'elev_ft'] = elevations[anchors]
            np.testing.assert_array_equal(np.flatnonzero(surface.elev_ft.notna()), anchors)
            common = {'event': event_name, 'environment': environment, 'site': site[scoring], 'mask': seed,
                      'block': [f'{int(x//10)}_{int(y//10)}' for x, y in xy[scoring]], 'reliefFt': regions[scoring]}
            original_random = np.random.default_rng(seed + 5435).permutation(candidates)
            estimates = {}
            for scenario in SCENARIOS:
                eligible, blocked, center = candidate_access(event_id, environment, seed, scenario, candidates, distances)
                assert len(eligible) >= count
                plans = make_plans(surface, distances, anchors, eligible, scoring, original_random, count)
                errors = {}
                for name, selected in plans.items():
                    assert len(set(selected)) == (0 if name == BASE else count)
                    assert set(selected) <= set(eligible)
                    measured = np.sort(np.r_[anchors, selected])
                    if not set(measured).isdisjoint(np.r_[blocked, scoring]):
                        raise AssertionError('Blocked or scored locations entered the survey')
                    key = tuple(measured)
                    if key not in estimates:
                        observed = surface.copy()
                        observed.loc[selected, 'elev_ft'] = elevations[selected]
                        np.testing.assert_array_equal(np.flatnonzero(observed.elev_ft.notna()), measured)
                        x = observation_features(observed, xy, distances, measured, scoring)
                        assert np.isnan(x['actual']).all()
                        estimates[key] = np.abs(elevations[scoring] - (x['default'] + model.predict(x['features'])))
                    errors[name] = estimates[key]
                rows[scenario].append(pd.DataFrame({**common, **errors}))
                budgets[scenario].append({'initial': len(anchors), 'originalCandidates': len(candidates),
                                          'blocked': len(blocked), 'available': len(eligible), 'visitBudget': count,
                                          'scored': len(scoring)})
                if case_seed == seed:
                    case[scenario].append({'environment': environment, 'visitBudget': count,
                                           'scoredCount': len(scoring), 'blockedSiteIds': [str(site[i]) for i in blocked],
                                           'clusterCenterId': str(site[center]) if center is not None else None,
                                           'clusterRadiusKm': float(distances[center, blocked].max()) if center is not None else None,
                                           'scoringSiteIds': [str(site[i]) for i in scoring],
                                           'methods': {name: {**stats(pd.DataFrame(errors))[name],
                                                              'chosenSiteIds': [str(site[i]) for i in visits]}
                                                       for name, visits in plans.items()}})
    for scenario in SCENARIOS:
        rows[scenario] = pd.concat(rows[scenario], ignore_index=True)
        budgets[scenario] = pd.DataFrame(budgets[scenario])
    pd.testing.assert_frame_equal(rows[OPEN], fixed_open[rows[OPEN].columns], check_exact=True)
    for scenario in (SCATTERED, CLUSTERED):
        pd.testing.assert_frame_equal(rows[scenario][COMMON], rows[OPEN][COMMON], check_exact=True)
        np.testing.assert_array_equal(rows[scenario][BASE], rows[OPEN][BASE])
        np.testing.assert_array_equal(budgets[scenario].visitBudget, budgets[OPEN].visitBudget)
        np.testing.assert_array_equal(budgets[scenario].scored, budgets[OPEN].scored)
    return rows, budgets, case


def penalty(closed, open_):
    from crew_replay import spatial_interval

    results = {}
    for name in READING_PLANS:
        paired = closed[['event', 'block', 'site']].assign(**{POLICY: open_[name].to_numpy(), 'closed': closed[name].to_numpy()})
        mae, bad = spatial_interval(paired, 'closed')
        results[name] = {'addedMaeFt': float((closed[name] - open_[name]).mean()),
                         'addedOverFiveRate': float(closed[name].gt(5).mean() - open_[name].gt(5).mean()),
                         'paired95AddedMaeFt': mae, 'paired95AddedOverFiveRate': bad}
    return results


def evaluate(panel, model, worked=None):
    from crew_replay import rows_for_storm, summarize

    rows, budgets = {scenario: [] for scenario in SCENARIOS}, {scenario: [] for scenario in SCENARIOS}
    for event_id, event in panel.groupby('event_id'):
        old, _ = rows_for_storm(event, model)
        selected = worked is not None and event_id == worked['eventId']
        result, counts, case = replay_event(event, model, old, worked['seed'] if selected else None)
        for scenario in SCENARIOS:
            rows[scenario].append(result[scenario])
            budgets[scenario].append(counts[scenario])
        if selected:
            worked['scenarios'] = case
    by_scenario = {name: pd.concat(parts, ignore_index=True) for name, parts in rows.items()}
    findings = {}
    initial = by_scenario[OPEN]
    steep = initial.environment.eq('Riverine') & initial.reliefFt.ge(35)
    for scenario, frame in by_scenario.items():
        counts = pd.concat(budgets[scenario], ignore_index=True)
        total = {str(key): int(value) for key, value in counts.sum().items()}
        primary = summarize(frame)
        primary['budget'] = {**total, 'actualBlockedFraction': total['blocked'] / total['originalCandidates'], 'cohortMasks': len(counts)}
        primary['highReliefRiverine'] = summarize(frame[steep])
        if scenario != OPEN:
            primary['penaltyAgainstSamePolicyNoClosure'] = penalty(frame, initial)
            primary['highReliefRiverine']['penaltyAgainstSamePolicyNoClosure'] = penalty(frame[steep], initial[steep])
        findings[scenario] = primary
    return findings


def main(package):
    sys.path.insert(0, str(package))
    from crew_replay import CASE_EVENT, CASE_SEED, KNOWN_FILES, digest, verified_fresh, write
    from map_model import TEST, TRAIN, fit_boosted, sample

    package_lock = json.loads((package / 'crew_input_lock.json').read_text())
    assert digest(package / 'crew_replay.py') == package_lock['analysisSha256']
    assert digest(package / 'survey_selection.json') == package_lock['priorSurveySelectionSha256']
    assert all(digest(package / name) == expected for name, expected in package_lock['priorInputs'].items())
    release_survey = json.loads((package / 'survey_selection.json').read_text())
    assert all(digest(package / name) == expected for name, expected in release_survey['freeze'].items())
    assert release_survey['selected']['tenPercent']['control'] == POLICY
    crew = json.loads((package / 'crew_results.json').read_text())
    assert crew['lockSha256'] == digest(package / 'crew_input_lock.json')
    frames = [pd.read_csv(package / name, keep_default_na=False) for name in KNOWN_FILES]
    fresh, checked = verified_fresh(package, pd.concat(frames, ignore_index=True), package, False)
    assert checked == package_lock['sources']
    original = frames[0]
    historical = pd.concat([original[original.event_id.isin(TEST)], *frames[1:3]], ignore_index=True)
    lock = {'protocolSha256': digest(HERE / 'ACCESS_STRESS_PROTOCOL.md'), 'codeSha256': digest(Path(__file__)),
            'originalCrewResultsSha256': digest(package / 'crew_results.json'),
            'originalCrewInputLockSha256': digest(package / 'crew_input_lock.json'),
            'blockedPercentTarget': 30, 'scenarios': list(SCENARIOS)}
    write(HERE / 'access_stress_lock.json', lock)
    worked = {'eventId': CASE_EVENT, 'seed': CASE_SEED}
    output = {'lockSha256': digest(HERE / 'access_stress_lock.json')}
    with threadpool_limits(4):
        model = fit_boosted(sample(original, TRAIN, 18))
        for name, frame in (('historical', historical), ('fresh', fresh)):
            print('Replaying', name, flush=True)
            outcome = evaluate(frame, model, worked if name == 'historical' else None)
            assert outcome[OPEN]['maskSitePairs'] == crew[name]['maskSitePairs']
            for method in (BASE, *READING_PLANS):
                np.testing.assert_allclose(outcome[OPEN]['methods'][method]['maeFt'], crew[name]['methods'][method]['maeFt'],
                                           rtol=0, atol=1e-10)
            output[name] = outcome
    write(HERE / 'access_stress_results.json', {**output, 'workedCase': worked})
    print('Access study finished', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=HERE)
    main(parser.parse_args().package.resolve())
