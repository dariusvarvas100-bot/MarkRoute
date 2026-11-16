"""Regression tests for the packaged MarkRoute data, exporter, and replay engine."""

from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE if (HERE / 'panel.csv').is_file() else HERE.parent
ROOT = Path(os.environ.get('MARKROUTE_ROOT', DEFAULT_ROOT)).resolve()
sys.path.insert(0, str(ROOT))

from demo_export import ARCHIVE_REPLAY_EVENT, ARCHIVE_REPLAY_MASKS, archival_summary, pack_trees, replay, review_exercise
from map_model import (DEV, OBSERVED, PEERS, SEED, STATIC_NAMES, TEST, TRAIN, episode,
                       fit_boosted, observation_features, sample, split_points, xy_km)


BASELINE = 'bounded local regression'
MODEL = 'gradient boost'
FIELDS = ('modelFt', 'baselineFt', 'startingFt', 'correctionFt')


def read_json(name):
    return json.loads((ROOT / name).read_text())


def replay_with_added_survey(frame, sites, index, model):
    active = np.array([i for i, site in enumerate(sites) if site['surveyed'] or i == index])
    queries = np.setdiff1d(np.arange(len(frame)), active)
    xy = xy_km(frame)
    distances = cdist(xy, xy)
    rows = [[None] * len(FIELDS) for _ in sites]
    for cohort in frame.groupby('environment').indices.values():
        targets = np.intersect1d(cohort, queries)
        if not len(targets):
            continue
        anchors = np.intersect1d(cohort, active)
        data = observation_features(frame, xy, distances, anchors if len(anchors) else active, targets)
        base = data['default']
        model_ft = base + model.predict(data['features'])
        bound = data['baseline'][:, STATIC_NAMES.index(BASELINE)]
        for target, values in zip(targets, zip(model_ft, bound, base, model_ft - base, strict=True), strict=True):
            rows[target] = list(values)
    return rows


def reading_counterfactuals(frame, sites, added_index, model):
    """Compute counterfactuals from independent Python masks over the same panel."""
    surveyed = np.array([site['surveyed'] for site in sites], dtype=bool)
    environments = frame.environment.to_numpy()
    xy = xy_km(frame)
    distances = cdist(xy, xy)
    cases = []
    for step in ('initial', 'afterAdd'):
        if step == 'afterAdd':
            surveyed[added_index] = True
        for target in np.flatnonzero(~surveyed):
            local = np.flatnonzero(surveyed & (environments == environments[target]))
            if len(local) >= 2:
                break
        else:
            raise AssertionError('No unmeasured target with two same-environment readings')

        closest = local[np.argsort(distances[target, local], kind='stable')][:min(4, len(local) - 1)]

        def estimate(available):
            # Blank unknown archive values before computing any feature, retaining
            # the full coordinate panel and original environment membership.
            copy = frame.copy()
            copy.loc[~copy.index.isin(available), 'elev_ft'] = np.nan
            features = observation_features(copy, xy, distances, available, np.array([target]))
            assert np.isnan(features['actual']).all()
            return float(features['default'][0] + model.predict(features['features'])[0])

        all_readings = estimate(local)
        for removed in closest:
            without = estimate(local[local != removed])
            cases.append({'step': step, 'target': int(target), 'removed': int(removed),
                          'fullFt': all_readings, 'withoutFt': without})
    return cases


class MarkRouteRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = pd.read_csv(ROOT / 'panel.csv', keep_default_na=False)
        cls.external = pd.read_csv(ROOT / 'external_panel.csv', keep_default_na=False)
        cls.remote = pd.read_csv(ROOT / 'remote_panel.csv', keep_default_na=False)
        cls.archival = pd.read_csv(ROOT / 'archival_panel.csv', keep_default_na=False)
        with threadpool_limits(4):
            cls.model = fit_boosted(sample(cls.panel, TRAIN, 18))

    def test_holdouts_and_archive_manifest_membership(self):
        groups = [set(ids) for ids in (TRAIN, DEV, TEST, self.external.event_id, self.remote.event_id, self.archival.event_id)]
        for left, right in combinations(groups, 2):
            self.assertTrue(left.isdisjoint(right))
        self.assertEqual(set(self.panel.event_id), set.union(*groups[:3]))
        for panel in (self.panel, self.external, self.remote, self.archival):
            self.assertFalse(panel.duplicated(['event_id', 'site_id']).any())
        for panel, file, key in ((self.external, 'external_manifest.json', 'events'),
                                 (self.remote, 'remote_manifest.json', 'eligibility')):
            retained = {event['id'] for event in read_json(file)[key] if not event.get('excluded')}
            self.assertEqual(set(panel.event_id), retained)
        previous = pd.concat([self.panel, self.external], ignore_index=True)
        self.assertTrue(set(self.remote.site_id).isdisjoint(previous.site_id))
        self.assertGreater(self.remote.distance_to_prior_km.min(), 100)
        self.assertTrue(set(self.archival.site_id).isdisjoint(pd.concat([previous, self.remote]).site_id))
        archive_manifest = read_json('archival_manifest.json')['events']
        self.assertEqual(set(self.archival.event_id), {event['id'] for event in archive_manifest if not event.get('excluded')})

    def test_reveal_masks_only_score_unseen_sites(self):
        for panel in (self.panel, self.external, self.remote, self.archival):
            for event_id, frame in panel.groupby('event_id'):
                frame = frame.reset_index(drop=True)
                for seed in (SEED, SEED + 15):
                    with self.subTest(event=event_id, seed=seed):
                        seen, unseen = split_points(frame, seed)
                        np.testing.assert_array_equal(split_points(frame, seed)[0], seen)
                        np.testing.assert_array_equal(np.union1d(seen, unseen), np.arange(len(frame)))
                        self.assertEqual(len(seen) + len(unseen), len(frame))
                        for cohort in frame.groupby('environment').indices.values():
                            expected = min(len(cohort) - 1, max(2, round(len(cohort) * OBSERVED)))
                            self.assertEqual(len(np.intersect1d(seen, cohort)), expected)
                        result = episode(frame, seed)
                        np.testing.assert_array_equal(np.sort(result['site']), np.sort(frame.iloc[unseen].site_id))
                        self.assertTrue(np.isfinite(result['baseline']).all())

    def test_hidden_elevations_cannot_change_inputs_or_prediction(self):
        for panel, ids in ((self.panel, TEST), (self.external, self.external.event_id.unique()),
                           (self.remote, self.remote.event_id.unique())):
            for event_id in ids:
                with self.subTest(event=event_id):
                    frame = panel.loc[panel.event_id.eq(event_id)].reset_index(drop=True)
                    _, unseen = split_points(frame, SEED)
                    changed = frame.copy()
                    changed.loc[unseen, 'elev_ft'] += 300_000 + np.arange(len(unseen))
                    original = episode(frame, SEED)
                    isolated = episode(changed, SEED)
                    self.assertTrue(np.all(original['actual'] != isolated['actual']))
                    for key in ('pair', 'mask', 'values', 'features', 'baseline', 'default'):
                        np.testing.assert_array_equal(original[key], isolated[key], err_msg=key)
                    np.testing.assert_array_equal(self.model.predict(original['features']),
                                                  self.model.predict(isolated['features']))

    def test_leave_one_out_review_excludes_own_elevation(self):
        frame = self.panel.loc[self.panel.event_id.eq(TEST[0])].reset_index(drop=True)
        seen, _ = split_points(frame, SEED)
        expected_peers = min(len(seen) - 1, PEERS)
        query = seen[:1]
        changed = frame.copy()
        changed.loc[query, 'elev_ft'] += 100_000
        xy = xy_km(frame)
        distances = cdist(xy, xy)
        reference = observation_features(frame, xy, distances, seen, query, exclude_self=True)
        isolated = observation_features(changed, xy, distances, seen, query, exclude_self=True)
        self.assertEqual(reference['mask'].sum(), expected_peers)
        self.assertNotEqual(reference['actual'][0], isolated['actual'][0])
        for key in ('values', 'features', 'baseline', 'default'):
            np.testing.assert_array_equal(reference[key], isolated[key], err_msg=key)

    def test_point_replay_reproduces_published_error_and_denominators(self):
        holdout = read_json('holdout.json')
        external = read_json('external_results.json')['additionalEventReplay']
        remote = read_json('remote_results.json')['all']
        archival = read_json('archival_results.json')['all']
        archive_events = {name: row['methods'] for name, row in archival['byEvent'].items()}
        groups = ((self.panel, TEST, holdout['evaluatedMaskLocationPairs'], holdout['results']['events']),
                  (self.external, self.external.event_id.unique(), external['maskSitePairs'], external['events']),
                  (self.remote, self.remote.event_id.unique(), remote['pooled']['maskSitePairs'], remote['events']),
                  (self.archival, self.archival.event_id.unique(), archival['maskSitePairs'], archive_events))
        with threadpool_limits(4):
            for panel, ids, count, events in groups:
                data = sample(panel, ids, 16)
                self.assertEqual(len(data['actual']), count)
                self.assertEqual(set(data['event']), set(events))
                predictions = {MODEL: self.model.predict(data['features']) + data['default'],
                               BASELINE: data['baseline'][:, STATIC_NAMES.index(BASELINE)]}
                for event, report in events.items():
                    belongs = data['event'] == event
                    for method, values in predictions.items():
                        with self.subTest(event=event, method=method):
                            errors = np.abs(values[belongs] - data['actual'][belongs])
                            self.assertAlmostEqual(float(errors.mean()), report[method]['maeFt'], places=9)
                            miss_rate = report[method].get('missOver5FtRate', report[method].get('gt5FtRate', report[method].get('overFiveFtRate')))
                            self.assertAlmostEqual(float((errors > 5).mean()), miss_rate, places=12)

    def test_shipped_export_and_javascript_replay_match_python(self):
        page = (ROOT / 'demo.html').read_text()
        engine = (ROOT / 'browser_engine.js').read_text()
        self.assertIn(engine, page)
        landing = (ROOT / 'index.html').read_text()
        for fragment in ('crew-planner', 'survey-import', 'visit-exercise'):
            self.assertIn(f'href="demo.html#{fragment}"', landing)
            self.assertIn(f'id="{fragment}"', page)
        data, _ = json.JSONDecoder().raw_decode(page.split('const MARKROUTE_DATA = ', 1)[1])
        self.assertEqual([event['id'] for event in data['events']], [*TEST, ARCHIVE_REPLAY_EVENT])
        self.assertEqual(data['summary'], read_json('holdout.json'))
        self.assertEqual(data['archive'], archival_summary())
        self.assertEqual(data['estimator'], pack_trees(self.model))
        exercise = data['reviewExercise']
        self.assertEqual(exercise, review_exercise())
        case = read_json('crew_results.json')['workedCase']
        montana = self.remote.loc[self.remote.event_id.eq(case['eventId'])].reset_index(drop=True)
        self.assertEqual([site['id'] for site in exercise['sites']], montana.site_id.astype(str).tolist())
        initial = [i for i, site in enumerate(exercise['sites']) if site['surveyed']]
        by_id = {site['id']: i for i, site in enumerate(exercise['sites'])}
        nearest = [by_id[site] for site in case['methods']['nearest']['chosenSiteIds'][:exercise['budget']]]
        targets = np.asarray(exercise['targetIndices'])
        xy = xy_km(montana)
        distances = cdist(xy, xy)
        checks = []
        for picked in (exercise['modelIndices'], nearest):
            anchors = np.asarray(sorted(initial + picked))
            self.assertFalse(set(anchors) & set(targets))
            blinded = montana.copy()
            blinded.loc[~blinded.index.isin(anchors), 'elev_ft'] = np.nan
            features = observation_features(blinded, xy, distances, anchors, targets)
            self.assertTrue(np.isnan(features['actual']).all())
            predicted = features['default'] + self.model.predict(features['features'])
            error = np.abs(predicted - montana.elev_ft.to_numpy()[targets])
            checks.append({'picks': picked, 'maeFt': float(error.mean()),
                           'largeMisses': int((error > 5).sum()), 'scored': len(targets)})
        exercise['pythonChecks'] = checks
        for event in data['events']:
            panel, count = (self.archival, ARCHIVE_REPLAY_MASKS) if event['id'] == ARCHIVE_REPLAY_EVENT else (self.panel, 16)
            frame = panel.loc[panel.event_id.eq(event['id'])].reset_index(drop=True)
            self.assertEqual([scenario['seed'] for scenario in event['scenarios']], list(range(SEED, SEED + count)))
            for scenario in event['scenarios']:
                expected = replay(frame, scenario['seed'], self.model)
                self.assertEqual(expected['counts'], scenario['counts'])
                self.assertEqual([site['id'] for site in scenario['sites']], [site['id'] for site in expected['sites']])
                self.assertEqual([site['surveyed'] for site in scenario['sites']], [site['surveyed'] for site in expected['sites']])
                view = [site for site in scenario['sites'] if not site['surveyed']]
                measured = [site for site in expected['sites'] if not site['surveyed']]
                np.testing.assert_allclose([[site[field] for field in FIELDS] for site in view],
                                           [[site[field] for field in FIELDS] for site in measured], rtol=0, atol=1e-9)
                index = next(i for i, site in enumerate(scenario['sites']) if not site['surveyed'])
                scenario['addedSurvey'] = index
                scenario['afterSurvey'] = replay_with_added_survey(frame, scenario['sites'], index, self.model)
                scenario['readingCounterfactuals'] = reading_counterfactuals(frame, scenario['sites'], index, self.model)

        runtime = os.environ.get('MARKROUTE_JS_RUNTIME') or shutil.which('node') or shutil.which('jsc')
        if not runtime:
            self.skipTest('Install Node.js or set MARKROUTE_JS_RUNTIME to verify numerical browser parity.')
        encoded = json.dumps(data, allow_nan=False, separators=(',', ':'))
        js_checks = (HERE / 'test_browser_engine.js').read_text()
        planner = (ROOT / 'crew_planner.js').read_text()
        self.assertIn(planner, page)
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'check.js'
            script.write_text(f'const MARKROUTE_DATA = {encoded};\n{engine}\n{planner}\n{js_checks}\n')
            result = subprocess.run([runtime, str(script)], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout.strip(), flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
