"""Build the portable MarkRoute replay with verified offline model parameters."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from map_model import ROOT, SEED, TRAIN, TEST, fit_boosted, observation_features, sample, split_points, xy_km


ARCHIVE_REPLAY_EVENT = 125  # June 2016 West Virginia flood
ARCHIVE_REPLAY_MASKS = 4


def pack_trees(model):
    trees=[]
    for iteration in model._predictors:
        tree=[]
        for node in iteration[0].nodes:
            if node['is_categorical']:
                raise ValueError('Browser encoder only supports the fitted numeric feature schema')
            if node['is_leaf']:
                tree.append([float(node['value'])])
            else:
                tree.append([int(node['feature_idx']),float(node['num_threshold']),
                             int(node['left']),int(node['right'])])
        trees.append(tree)
    return {'intercept':float(model._baseline_prediction[0,0]),'trees':trees}


def replay(frame,seed,model):
    xy=xy_km(frame)
    distances=cdist(xy,xy)
    seen,queries=split_points(frame,seed)
    points=[]
    reviewed=set(seen)
    estimates={}
    for candidates in frame.groupby('environment').indices.values():
        wanted=np.intersect1d(queries,candidates)
        neighbors=np.intersect1d(seen,candidates)
        if not len(wanted):
            continue
        inputs=observation_features(frame,xy,distances,neighbors if len(neighbors) else seen,wanted)
        baseline=inputs['baseline'][:,3]
        predicted=inputs['default']+model.predict(inputs['features'])
        estimates.update(zip(wanted,zip(predicted,baseline,inputs['default'],strict=True),strict=True))
    for idx,row in enumerate(frame.itertuples(index=False)):
        model_ft,base_ft,start_ft=estimates.get(idx,(None,None,None))
        points.append({'id':str(row.site_id),'lat':row.latitude,'lon':row.longitude,
                       'environment':row.environment,'waterbody':row.waterbody,
                       'surveyed':idx in reviewed,'actualFt':row.elev_ft,
                       'modelFt':model_ft,'baselineFt':base_ft,'startingFt':start_ft,
                       'correctionFt':None if start_ft is None else model_ft-start_ft,
                       'groundFt':row.terrain_median_ft,
                       'pointFt':row.terrain_ft,'reliefFt':row.relief_ft})
    def error(key):
        return float(np.mean([abs(point[key]-point['actualFt']) for point in points if not point['surveyed']]))
    return {'seed':seed,'sites':points,'counts':{'sites':len(points),'initial':len(seen),'unseen':len(queries)},
            'mae':{'model':error('modelFt'),'baseline':error('baselineFt')}}


def archival_summary():
    result_path = ROOT / 'archival_results.json'
    if not result_path.is_file():
        return None
    from archival_replay import BASELINE, ESTIMATOR, KRIGING, RELIEF_FT

    report = json.loads(result_path.read_text())
    panel = pd.read_csv(ROOT / 'archival_panel.csv')
    medians = panel.groupby('event_name').relief_ft.median()
    overall = report['all']
    steep = report['highReliefPrimary']

    def mae(methods):
        return {'markroute': methods[ESTIMATOR]['maeFt'],
                'bounded': methods[BASELINE]['maeFt'], 'kriging': methods[KRIGING]['maeFt']}

    def pooled(cohort):
        return {'siteCount': cohort['uniqueEventSites'], 'pairCount': cohort['maskSitePairs'],
                'mae': mae(cohort['methods']), 'comparisonModel': 'kriging',
                'intervalFt': cohort['comparisons'][KRIGING]['paired95IntervalFt']}

    return {
        'title': 'Nine additional river floods',
        'geographyNote': 'Each flood was withheld from training and development. Exact prior site IDs were '
                         'excluded. Some sites are near earlier survey locations.',
        'primary': {**pooled(overall), 'label': 'All nine additional river floods'},
        'steep': {**pooled(steep), 'cutoffFt': RELIEF_FT,
                  'overFiveFtRate': steep['methods'][ESTIMATOR]['overFiveFtRate']} if steep else None,
        'events': [{'label': name, 'siteCount': row['sites'], 'pairCount': row['maskSitePairs'],
                    'medianReliefFt': float(medians[name]), 'mae': mae(row['methods'])}
                   for name, row in overall['byEvent'].items()],
    }


def crew_summary():
    path = ROOT / 'crew_results.json'
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    report = result['historical']
    methods = report['methods']
    coverage = methods['terrain coverage']['maeFt']
    nearest = methods['nearest']['maeFt']
    example = result['workedCase']
    selected = example['methods']
    terrain_ids = ', '.join(selected['terrain coverage']['chosenSiteIds'][:3])
    nearest_ids = ', '.join(selected['nearest']['chosenSiteIds'][:3])
    return {
        'summary': f'Same number of visits across 13 archived floods: {nearest:.2f} ft average error '
                   f'for nearest-first, {coverage:.2f} ft for terrain coverage.',
        'example': f'June 2022 Montana flood: start from {example["initialCount"]} known readings and take '
                   f'{example["visitBudget"]} more. Terrain coverage starts with marks {terrain_ids}; nearest-first '
                   f'starts with {nearest_ids}. On the same {example["scoredCount"]} untouched marks, '
                   f'average error was {selected["terrain coverage"]["maeFt"]:.2f} ft with terrain coverage '
                   f'and {selected["nearest"]["maeFt"]:.2f} ft with nearest-first.',
    }


def review_exercise():
    results = ROOT / 'crew_results.json'
    if not results.is_file():
        return None
    case = json.loads(results.read_text())['workedCase']
    if len(case['cohorts']) != 1:
        raise ValueError('The review exercise uses the single-environment Montana split')
    cohort = case['cohorts'][0]
    remote = pd.read_csv(ROOT / 'remote_panel.csv', keep_default_na=False)
    points = remote.loc[remote.event_id.eq(case['eventId'])]
    initial = set(cohort['initialSiteIds'])
    sites = [{'id': str(row.site_id), 'lat': row.latitude, 'lon': row.longitude,
              'environment': row.environment, 'waterbody': row.waterbody,
              'groundFt': row.terrain_median_ft, 'pointFt': row.terrain_ft,
              'reliefFt': row.relief_ft, 'actualFt': row.elev_ft,
              'surveyed': str(row.site_id) in initial} for row in points.itertuples(index=False)]
    by_id = {site['id']: index for index, site in enumerate(sites)}
    if len(by_id) != case['siteCount']:
        raise ValueError('The review exercise must match the locked Montana panel')
    candidates = [by_id[site] for site in cohort['candidateSiteIds']]
    targets = [by_id[site] for site in cohort['scoringSiteIds']]
    model = [by_id[site] for site in case['methods']['terrain coverage']['chosenSiteIds'][:3]]
    if set(candidates) & set(targets) or any(sites[i]['surveyed'] for i in candidates + targets):
        raise ValueError('The review exercise cannot share measured, candidate, or scoring sites')
    if len(model) != 3 or not set(model).issubset(candidates):
        raise ValueError('The review exercise requires three locked eligible model picks')
    return {'title': 'June 2022 Montana flood', 'budget': 3, 'sites': sites,
            'candidateIndices': candidates, 'targetIndices': targets, 'modelIndices': model}


def build():
    frame=pd.read_csv(ROOT/'panel.csv',keep_default_na=False)
    model=fit_boosted(sample(frame,TRAIN,18))
    events=[]
    for id_ in TEST:
        points=frame[frame.event_id.eq(id_)].reset_index(drop=True)
        events.append({'id':id_,'name':points.event_name.iat[0],
                       'scenarios':[replay(points,seed,model) for seed in range(SEED,SEED+16)]})
    archive_path = ROOT / 'archival_panel.csv'
    if archive_path.is_file():
        archive = pd.read_csv(archive_path, keep_default_na=False)
        west_virginia = archive.loc[archive.event_id.eq(ARCHIVE_REPLAY_EVENT)].reset_index(drop=True)
        if not west_virginia.empty:
            events.append({'id': ARCHIVE_REPLAY_EVENT, 'name': 'June 2016 West Virginia flood',
                           'scenarios': [replay(west_virginia, seed, model)
                                         for seed in range(SEED, SEED + ARCHIVE_REPLAY_MASKS)]})
    payload = {'summary': json.loads((ROOT/'holdout.json').read_text()), 'events': events,
               'estimator': pack_trees(model)}
    external = ROOT/'external_results.json'
    if external.is_file():
        payload['external'] = json.loads(external.read_text())
    remote = ROOT/'remote_results.json'
    if remote.is_file():
        report = json.loads(remote.read_text())
        values = report['all']['pooled']
        payload['remote'] = {
            'siteCount': sum(row['sites'] for row in report['sitesAndDistances'].values()),
            'eventCount': len(report['all']['events']),
            'baseline': values['bounded local regression']['maeFt'],
            'model': values['gradient boost']['maeFt'],
        }
    review = ROOT/'review_results.json'
    if review.is_file():
        report = json.loads(review.read_text())
        stats = report['results']['all test']['methods']['model disagreement']
        pooled = stats['pooled']
        payload['review'] = {'budget': report['budgetFraction'], 'caught': pooled['largeErrorsFlagged'],
                             'totalLargeErrors': pooled['largeErrorPairs'], 'eventCount': len(stats['events'])}
    robustness = ROOT/'robustness_results.json'
    if robustness.is_file():
        report = json.loads(robustness.read_text())
        def paired_metrics(phase):
            return {'model': phase['pooled']['gradient boost']['maeFt'],
                    'baseline': phase['pooled']['bounded local regression']['maeFt']}
        budgets = report['nominalBudgetsPercent']
        gaps = report['geographicGap']
        dispersed = paired_metrics(gaps['dispersed'])
        missing = paired_metrics(gaps['missingCluster'])
        payload['robustness'] = {'budgets': budgets,
                                 'points': [{'percent': budget, **paired_metrics(report['commonTargets'][str(budget)])}
                                            for budget in budgets],
                                 'gap': {'modelDispersed': dispersed['model'], 'modelMissing': missing['model'],
                                         'baselineDispersed': dispersed['baseline'], 'baselineMissing': missing['baseline']}}
    archival = archival_summary()
    if archival is not None:
        payload['archive'] = archival
    crew = crew_summary()
    if crew is not None:
        payload['crew'] = crew
    exercise = review_exercise()
    if exercise is not None:
        payload['reviewExercise'] = exercise
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',type=Path,help='Write the packed benchmark for browser parity checks')
    parser.add_argument('--output',type=Path,default=ROOT/'demo.html')
    args=parser.parse_args()
    payload=build()
    serialized=json.dumps(payload,ensure_ascii=True,allow_nan=False,separators=(',',':'))
    if args.json:
        args.json.write_text(serialized+'\n')
    template=ROOT/'demo_template.html'
    if template.exists():
        html=template.read_text()
        markers=('/*__MARKROUTE_DATA__*/','/*__MARKROUTE_ENGINE__*/')
        if any(html.count(marker)!=1 for marker in markers):
            raise ValueError('The page template must declare one marker each for model data and the engine')
        data='const MARKROUTE_DATA = '+serialized.replace('<','\\u003c')+';'
        html=html.replace(markers[0],data).replace(markers[1],(ROOT/'browser_engine.js').read_text())
        crew_marker = '/*__MARKROUTE_CREW__*/'
        if crew_marker in html:
            if html.count(crew_marker) != 1:
                raise ValueError('The page template must declare one visit planner marker')
            html = html.replace(crew_marker, (ROOT / 'crew_planner.js').read_text())
        args.output.write_text(html)


if __name__=='__main__':
    main()
