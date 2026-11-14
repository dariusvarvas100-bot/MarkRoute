"""Compare local, measured-error blending against frozen MarkRoute predictions."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from map_model import DEV, ROOT, SEED, TEST, TRAIN, bootstrap, fit_boosted
from map_model import metrics, observation_features, sample, split_points, xy_km


def prediction_matrix(features, model):
    gradient=features['default'] + model.predict(features['features'])
    return np.c_[gradient, features['baseline'][:, 2], features['baseline'][:, 3]]


def local_scores(frame, xy, d, context, queries, model):
    if len(context)<2:
        return np.full((len(queries),3),np.inf)
    inner=observation_features(frame,xy,d,context,context,exclude_self=True)
    misses=np.abs(inner['actual'][:,None]-prediction_matrix(inner,model))
    ground=frame.terrain_median_ft.to_numpy()
    water=frame.waterbody.to_numpy()
    same=(water[queries,None]==water[context]) & (water[queries,None]!='')
    dh=np.abs(ground[queries,None]-ground[context])
    w=(1+same)/(0.25+d[np.ix_(queries,context)])**1.5/(1+dh/30)
    w/=w.sum(axis=1,keepdims=True)
    return w@misses


def episode(frame, seed, model):
    frame=frame.reset_index(drop=True)
    xy=xy_km(frame)
    d=cdist(xy,xy)
    seen,withheld=split_points(frame,seed)
    result=[]
    for ids in frame.groupby('environment').indices.values():
        context=np.intersect1d(seen,ids)
        queries=np.intersect1d(withheld,ids)
        if not len(queries):
            continue
        if not len(context):
            context=seen
        features=observation_features(frame,xy,d,context,queries)
        result.append({'predictions':prediction_matrix(features,model),
                       'cross_checks':local_scores(frame,xy,d,context,queries,model),
                       'actual':features['actual'],'event':frame.event_name.to_numpy()[queries],
                       'environment':frame.environment.to_numpy()[queries],
                       'block':np.asarray([f'{int(x//10)}_{int(y//10)}' for x,y in xy[queries]])})
    return join(result)


def join(items):
    return {key:np.concatenate([item[key] for item in items]) for key in items[0]}


def group(panel, storms, model):
    return join([episode(panel[panel.event_id==storm],seed,model)
                 for storm in storms for seed in range(SEED,SEED+16)])


def blended(data, margin, strength):
    prediction=data['predictions']
    costs=data['cross_checks']
    usable=np.isfinite(costs).all(axis=1)
    scale=1.+costs[usable,0]
    relatives=(costs[usable,0,None]-costs[usable,1:])/scale[:,None]
    # Historical generalization remains the default. An alternative gets
    # weight only when reconstruction of actual same-storm readings favors it.
    preference=np.maximum(relatives-margin,0)
    ratios=np.c_[np.ones(usable.sum()),strength*preference]
    output=prediction[:,0].copy()
    output[usable]=(ratios*prediction[usable]).sum(1)/ratios.sum(1)
    return output


def run(stage):
    panel=pd.read_csv(ROOT/'panel.csv',keep_default_na=False)
    extra=pd.read_csv(ROOT/'external_panel.csv',keep_default_na=False)
    model=fit_boosted(sample(panel,TRAIN,18))
    settings=[(margin,strength) for margin in (.10,.25,.50) for strength in (1,3,8)]
    saved=ROOT/'adaptive_selection.json'
    if stage=='validation':
        dev=group(panel,DEV,model)
        candidates=[(float(np.mean(np.abs(blended(dev,*setting)-dev['actual']))),setting)
                    for setting in settings]
        best=min(candidates)
        saved.write_text(json.dumps({'selected':best[1],'validationMaeFt':best[0],'choices':candidates},indent=2)+'\n')
        print(saved.read_text(),flush=True)
        return
    locked=json.loads(saved.read_text())
    cohorts={'primary':group(panel,TEST,model),'external':group(extra,extra.event_id.unique(),model)}
    output={}
    for key,data in cohorts.items():
        original=data['predictions'][:,0]
        candidate=blended(data,*locked['selected'])
        output[key]={'scores':metrics({'MarkRoute':original,'adaptive blend':candidate},data),
                     'absoluteGainPaired95Ft':bootstrap(data,candidate,original),
                     'changedPredictions':int(np.count_nonzero(np.abs(candidate-original)>1e-8))}
    dest=ROOT/'adaptive_results.json'
    dest.write_text(json.dumps(output,indent=2)+'\n')
    print(dest.read_text(),flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('validation','test'))
    with threadpool_limits(limits=4):
        run(parser.parse_args().stage)
