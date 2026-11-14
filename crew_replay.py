"""Same-budget retrospective interventions on the set of measured flood marks."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
NAMES = ('no extra', 'terrain coverage', 'nearest', 'farthest', 'random')
CONTROLS = ('nearest', 'farthest', 'random')
KNOWN_FILES = ('panel.csv', 'external_panel.csv', 'remote_panel.csv', 'archival_panel.csv')
CASE_EVENT, CASE_SEED = 319, 7162
FRESH_CASES = ('AR_LittleMissouri_2010', 'NH_ColdRiver_2005')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def spherical(frame):
    lat, lon = np.radians(frame[['latitude', 'longitude']].to_numpy().T)
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def verified_fresh(directory, prior, package, verify_reports):
    rows, sources = [], {}
    coordinates = prior[['latitude', 'longitude']].to_numpy()
    known_cells = set(map(tuple, np.floor(coordinates * 600).astype(int)))
    known_tree = cKDTree(spherical(prior))
    for name in FRESH_CASES:
        file = directory / f'{name}_panel.csv'
        source_lock = directory / f'{name}_source_lock.json'
        metadata = json.loads(source_lock.read_text())
        assert digest(file) == metadata['preparedPanelSha256']
        if verify_reports:
            assert digest(directory / f'{name}.pdf') == metadata['sourcePdfSha256']
        assert all(digest(package / path) == known for path, known in metadata['preexistingPanelHashes'].items())
        panel = pd.read_csv(file, keep_default_na=False)
        assert len(panel) == metadata['gridSites'] and panel.event_id.nunique() == 1
        assert set(panel.datum) == {'NAVD88'} and set(panel.unit) == {'ft'} and set(panel.environment) == {'Riverine'}
        assert panel.site_id.is_unique and set(panel.site_id).isdisjoint(prior.site_id)
        assert set(panel.event_id).isdisjoint(prior.event_id)
        assert np.isfinite(panel[['latitude', 'longitude', 'terrain_ft', 'terrain_median_ft', 'relief_ft']]).all(axis=None)
        grid = set(map(tuple, np.floor(panel[['latitude', 'longitude']].to_numpy() * 600).astype(int)))
        assert len(grid) == len(panel)
        chord = known_tree.query(spherical(panel))[0]
        distance = 2 * 6371.0088 * np.arcsin(np.clip(chord / 2, 0, 1))
        sources[name] = {'eventId': panel.event_id.iloc[0], 'source': metadata['source'],
                         'panelSha256': digest(file), 'sourceLockSha256': digest(source_lock),
                         'fieldReportSha256': metadata['sourcePdfSha256'], 'surveySites': len(panel),
                         'steepRiverineSites': int(panel.relief_ft.ge(35).sum()),
                         'nearestPriorKm': float(distance.min()), 'medianNearestPriorKm': float(np.median(distance)),
                         'priorGridCellsShared': len(grid & known_cells), 'exactPriorSiteIdsShared': 0,
                         'horizontalDatumNote': metadata.get('horizontal')}
        rows.append(panel)
    return pd.concat(rows, ignore_index=True), sources


def rows_for_storm(frame, model, case=None):
    from map_model import observation_features, xy_km
    from survey_decision import MASKS, greedy_coverage, kernel, partition

    frame = frame.reset_index(drop=True)
    xy = xy_km(frame)
    distances = cdist(xy, xy)
    actual = frame.elev_ft.to_numpy()
    site_ids = frame.site_id.to_numpy()
    labels = frame.event_name.to_numpy()
    environment = frame.environment.to_numpy()
    relief = frame.relief_ft.to_numpy()
    replays = []
    budgets = []
    case_rows = []
    for seed in MASKS:
        for anchors, candidates, scoring, count in partition(frame, seed):
            # Choose every extra visit before making any candidate elevation available.
            hidden = np.r_[candidates, scoring]
            surface = frame.assign(elev_ft=np.nan)
            surface.loc[anchors, 'elev_ft'] = actual[anchors]
            assert surface.elev_ft.notna().to_numpy().nonzero()[0].tolist() == sorted(anchors)
            terrain = kernel(surface, distances, anchors, hidden, .75)
            farthest, coverage = greedy_coverage(distances, terrain, anchors, candidates, scoring, count)
            neighborhood = distances[np.ix_(candidates, anchors)].min(axis=1)
            nearest = candidates[np.argsort(neighborhood, kind='stable')[:count]]
            random = np.random.default_rng(seed + 5435).permutation(candidates)[:count]
            selections = {'no extra': np.array([], dtype=int), 'terrain coverage': coverage,
                          'nearest': nearest, 'farthest': farthest, 'random': random}
            results = {}
            for name, selected in selections.items():
                assert len(set(selected)) == (0 if name == 'no extra' else count)
                assert set(selected) <= set(candidates) and not (set(selected) & set(scoring))
                known = np.sort(np.r_[anchors, selected])
                repl = surface.copy()
                repl.loc[selected, 'elev_ft'] = actual[selected]
                factors = observation_features(repl, xy, distances, known, scoring)
                assert np.isnan(factors['actual']).all(), 'scoring elevations must stay unavailable'
                predicted = factors['default'] + model.predict(factors['features'])
                results[name] = np.abs(actual[scoring] - predicted)
            replays.append(pd.DataFrame({'event': labels[scoring], 'site': site_ids[scoring], 'mask': seed,
                                         'environment': environment[scoring], 'reliefFt': relief[scoring],
                                         'block': [f'{int(x//10)}_{int(y//10)}' for x, y in xy[scoring]], **results}))
            budgets.append({'event': labels[scoring][0], 'anchors': len(anchors), 'candidates': len(candidates),
                            'scored': len(scoring), 'visits': count, 'eligibleCohortSites': len(anchors) + len(hidden)})
            if case and seed == case['seed']:
                row = {'environment': environment[scoring][0], 'anchorIndices': anchors, 'candidateIndices': candidates,
                       'scoringIndices': scoring, 'budget': count, 'selected': selections}
                case['cohorts'].append(row)
                case_rows.append(pd.DataFrame(results))
    if case:
        all_case = pd.concat(case_rows, ignore_index=True)
        initial = set(int(index) for cohort in case['cohorts'] for index in cohort['anchorIndices'])
        candidate = set(int(index) for cohort in case['cohorts'] for index in cohort['candidateIndices'])
        targets = set(int(index) for cohort in case['cohorts'] for index in cohort['scoringIndices'])
        methods = stats(all_case)
        for method in NAMES:
            choices = [int(index) for cohort in case['cohorts'] for index in cohort['selected'][method]]
            methods[method]['chosenSiteIds'] = [str(site_ids[i]) for i in choices]
        case.update({'eventName': labels[0], 'siteCount': len(frame), 'initialCount': len(initial),
                     'visitBudget': sum(cohort['budget'] for cohort in case['cohorts']), 'scoredCount': len(targets),
                     'initialSiteIds': [str(site_ids[i]) for i in sorted(initial)],
                     'candidateSiteIds': [str(site_ids[i]) for i in sorted(candidate)],
                     'scoringSiteIds': [str(site_ids[i]) for i in sorted(targets)], 'methods': methods})
        case['cohorts'] = [{'environment': row['environment'], 'visitBudget': row['budget'],
                            'initialCount': len(row['anchorIndices']), 'candidateCount': len(row['candidateIndices']),
                            'scoredCount': len(row['scoringIndices']),
                            'initialSiteIds': [str(site_ids[i]) for i in row['anchorIndices']],
                            'candidateSiteIds': [str(site_ids[i]) for i in row['candidateIndices']],
                            'scoringSiteIds': [str(site_ids[i]) for i in row['scoringIndices']]} for row in case['cohorts']]
    return pd.concat(replays, ignore_index=True), pd.DataFrame(budgets)


def stats(frame):
    return {name: {'maeFt': float(frame[name].mean()), 'overFive': int(frame[name].gt(5).sum()),
                   'overFiveRate': float(frame[name].gt(5).mean())} for name in NAMES}


def spatial_interval(frame, reference):
    rng = np.random.default_rng(CASE_SEED)
    values = []
    for _, event in frame.groupby('event', sort=True):
        gain = event[reference] - event['terrain coverage']
        fewer = event[reference].gt(5).astype(int) - event['terrain coverage'].gt(5).astype(int)
        group = event.assign(gain=gain, fewer=fewer).groupby('block').agg({'gain': 'sum', 'fewer': 'sum', 'site': 'size'})
        draw = rng.integers(len(group), size=(4000, len(group)))
        values.append(group.to_numpy()[draw].sum(axis=1))
    combined = np.sum(values, axis=0)
    return np.quantile(combined[:, :2] / combined[:, 2, None], [.025, .975], axis=0).T.tolist()


def summarize(frame):
    methods = stats(frame)
    events = {label: {'uniqueSites': int(group.site.nunique()), 'maskSitePairs': len(group),
                      'spatialBlocks': int(group.block.nunique()), 'methods': stats(group)}
              for label, group in frame.groupby('event')}
    gain_vs = {}
    for method in ('no extra', *CONTROLS):
        delta = methods[method]['maeFt'] - methods['terrain coverage']['maeFt']
        mae_interval, miss_interval = spatial_interval(frame, method)
        gain_vs[method] = {'maeReductionFt': delta, 'relativeMaeReduction': delta / methods[method]['maeFt'],
                           'overFiveRateReduction': methods[method]['overFiveRate'] - methods['terrain coverage']['overFiveRate'],
                           'paired95WithinEventBlocksFt': mae_interval, 'overFiveRatePaired95': miss_interval,
                           'eventWins': int(sum(e['methods'][method]['maeFt'] > e['methods']['terrain coverage']['maeFt']
                                                for e in events.values()))}
    return {'eventCount': len(events), 'uniqueEventSites': sum(e['uniqueSites'] for e in events.values()),
            'maskSitePairs': len(frame), 'methods': methods, 'comparisons': gain_vs, 'events': events}


def evaluate(panel, model, case=None):
    results, budgets = zip(*(rows_for_storm(rows, model, case if (case and code == case['eventId']) else None)
                             for code, rows in panel.groupby('event_id')))
    frame = pd.concat(results, ignore_index=True)
    budget = pd.concat(budgets, ignore_index=True)
    steep = frame[frame.environment.eq('Riverine') & frame.reliefFt.ge(35)]
    report = summarize(frame)
    report['budget'] = {key: int(budget[key].sum()) for key in ('anchors', 'candidates', 'scored', 'visits', 'eligibleCohortSites')}
    report['highReliefRiverine'] = summarize(steep)
    return report


def main(package, fresh, verify_reports):
    sys.path.insert(0, str(package))
    from map_model import TEST, TRAIN, fit_boosted, sample

    prior_frames = [pd.read_csv(package / name, keep_default_na=False) for name in KNOWN_FILES]
    prior = pd.concat(prior_frames, ignore_index=True)
    original = prior_frames[0]
    locked = json.loads((package / 'survey_selection.json').read_text())
    assert all(digest(package / path) == expected for path, expected in locked['freeze'].items())
    assert locked['selected']['tenPercent']['control'] == 'terrain coverage'
    fresh_panel, provenance = verified_fresh(fresh, prior, package, verify_reports) if fresh else (None, {})
    lock = {'protocolSha256': digest(HERE / 'CREW_REPLAY_PROTOCOL.md'), 'analysisSha256': digest(Path(__file__)),
            'priorInputs': {name: digest(package / name) for name in KNOWN_FILES},
            'priorSurveySelectionSha256': digest(package / 'survey_selection.json'), 'sources': provenance,
            'illustration': {'eventId': CASE_EVENT, 'seed': CASE_SEED}}
    write(HERE / 'crew_input_lock.json', lock)
    historical = pd.concat([original[original.event_id.isin(TEST)], *prior_frames[1:3]], ignore_index=True)
    worked = {'eventId': CASE_EVENT, 'seed': CASE_SEED, 'cohorts': []}
    print('Running frozen historical cohort', flush=True)
    with threadpool_limits(4):
        model = fit_boosted(sample(original, TRAIN, 18))
        report = evaluate(historical, model, worked)
        old = json.loads((package / 'survey_decision_results.json').read_text())['tenPercent']
        assert report['maskSitePairs'] == old['scoredEpisodes']
        for method in ('no extra', 'terrain coverage', 'farthest', 'random'):
            np.testing.assert_allclose(report['methods'][method]['maeFt'], old['methods'][method]['maeFt'], rtol=0, atol=1e-10)
            np.testing.assert_allclose(report['methods'][method]['overFiveRate'], old['methods'][method]['missOver5Ft'], rtol=0, atol=1e-10)
        output = {'historical': report, 'workedCase': worked, 'lockSha256': digest(HERE / 'crew_input_lock.json')}
        write(HERE / 'crew_results.json', output)
        if fresh_panel is not None:
            print('Historical reproduced; running frozen official report cohort', flush=True)
            output['fresh'] = evaluate(fresh_panel, model)
        write(HERE / 'crew_results.json', output)
    print('Crew replay complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=HERE, help='Directory containing the original MarkRoute package.')
    parser.add_argument('--fresh', type=Path, help='Directory containing the two prepared field panels and source locks.')
    parser.add_argument('--verify-reports', action='store_true', help='Also verify the original reports as local PDFs.')
    args = parser.parse_args()
    package_dir = args.package.resolve()
    field_panels = args.fresh.resolve() if args.fresh else (
        package_dir if all((package_dir / f'{name}_panel.csv').is_file() for name in FRESH_CASES) else None)
    if args.verify_reports and field_panels is None:
        parser.error('--verify-reports requires the fresh panels and their original PDFs')
    main(package_dir, field_panels, args.verify_reports)
