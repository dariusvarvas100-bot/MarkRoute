"""Build the portable MarkRoute replay with verified offline model parameters."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from map_model import ROOT, SEED, TRAIN, TEST, fit_boosted, observation_features, sample, split_points, xy_km


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


def build():
    frame=pd.read_csv(ROOT/'panel.csv',keep_default_na=False)
    model=fit_boosted(sample(frame,TRAIN,18))
    events=[]
    for id_ in TEST:
        points=frame[frame.event_id.eq(id_)].reset_index(drop=True)
        events.append({'id':id_,'name':points.event_name.iat[0],
                       'scenarios':[replay(points,seed,model) for seed in range(SEED,SEED+16)]})
    payload = {'summary': json.loads((ROOT/'holdout.json').read_text()), 'events': events,
               'estimator': pack_trees(model)}
    external = ROOT/'external_results.json'
    if external.is_file():
        payload['external'] = json.loads(external.read_text())
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
        args.output.write_text(html)


if __name__=='__main__':
    main()
