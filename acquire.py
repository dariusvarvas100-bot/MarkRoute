"""Replay extra USGS survey visits with equal measurement budgets."""
import argparse
import json
from math import ceil

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingRegressor
import torch

from map_model import ROOT, TEST, TRAIN, DEV, SEED, WaterAttention, observation_features, scores
from map_model import mae, split_points, sample, xy_km

REPLAYS=16
METHODS=('farthest from surveys','largest expected error','terrain coverage','targeted surveys')


def predict_error(data, error_model):
    return np.maximum(error_model.predict(data['features']),.05)


def choose(method, probabilities, uncertainty, distance, priority, candidate_distances, required):
    picked=[]
    seen=np.zeros(probabilities.shape[1])
    nearest=distance.copy()
    for _ in range(required):
        if method=='farthest from surveys':
            value=nearest.copy()
        elif method=='largest expected error':
            value=uncertainty.copy()
        else:
            marginal=np.maximum(probabilities-seen[None,:],0)
            if method=='targeted surveys':
                value=(marginal@priority)*np.sqrt(uncertainty)
            else:
                value=marginal.sum(1)
        value[picked]=-np.inf
        index=int(value.argmax())
        picked.append(index)
        seen=np.maximum(seen,probabilities[index])
        if method=='farthest from surveys':
            nearest=np.minimum(nearest,candidate_distances[index])
    return picked


def trial(model,error_model,frame,seed):
    frame=frame.reset_index(drop=True)
    xy=xy_km(frame)
    d=cdist(xy,xy)
    known,remainder=split_points(frame,seed)
    rng=np.random.default_rng(SEED+2000+seed)
    rows=[]
    for environment,ids in frame.groupby('environment').indices.items():
        seen=np.intersect1d(known,ids)
        unknown=np.intersect1d(remainder,ids)
        if len(seen)<1 or len(unknown)<8:
            continue
        rng.shuffle(unknown)
        candidates,scoring=np.split(unknown,[len(unknown)//2])
        k=min(len(candidates),max(1,ceil(len(ids)*.10)))
        queries=np.r_[candidates,scoring]
        first=observation_features(frame,xy,d,seen,queries)
        start_pred=scores(model,first)[len(candidates):]
        risk=predict_error(first,error_model)
        target_grid=frame.terrain_median_ft.to_numpy()
        delta=np.abs(target_grid[candidates,None]-target_grid[scoring])
        dist=d[np.ix_(candidates,scoring)]
        base_distance=d[np.ix_(scoring,seen)].min(1)
        density=np.clip(base_distance,.4,30)
        similarity=np.exp(-dist/density[None,:])/(1+delta/35)
        matches=frame.waterbody.to_numpy()
        same=(matches[candidates,None]==matches[scoring]) & (matches[candidates,None]!='')
        similarity=similarity*(.8+.2*same)
        result={}
        candidate_distances=d[np.ix_(candidates,candidates)]
        for method in METHODS:
            selected=choose(method,similarity,risk[:len(candidates)],
                            d[np.ix_(candidates,seen)].min(1),risk[len(candidates):],candidate_distances,k)
            updated=observation_features(frame,xy,d,np.r_[seen,candidates[selected]],scoring)
            values=scores(model,updated)
            result[method]=mae(updated['actual'],values)
        rows.append({'event':frame.event_name.iat[0],'environment':environment,'seed':seed,
                     'sites':len(scoring),'slots':k,'start':mae(first['actual'][len(candidates):],start_pred),
                     **result})
    return rows


def report(rows):
    frame=pd.DataFrame(rows)
    def summarize(data):
        return {key:float(np.average(data[key],weights=data.sites)) for key in ('start',*METHODS)}
    return {'pooled':summarize(frame),'perEvent':{e:summarize(part)
            for e,part in frame.groupby('event')},'episodes':len(frame)}


def main(stage):
    torch.set_num_threads(4)
    model=WaterAttention()
    model.load_state_dict(torch.load(ROOT/'water_attention.pth',weights_only=True))
    panel=pd.read_csv(ROOT/'panel.csv',keep_default_na=False)
    teaching=sample(panel,TRAIN,13)
    y=np.abs(teaching['actual']-scores(model,teaching))
    forecaster=HistGradientBoostingRegressor(loss='absolute_error',max_iter=110,max_leaf_nodes=14,
                                               l2_regularization=3,random_state=SEED)
    forecaster.fit(teaching['features'],y)
    rows=[]
    for event in (DEV if stage=='validation' else TEST):
        terrain=panel[panel.event_id==event]
        for seed in range(SEED+30,SEED+30+REPLAYS):
            rows.extend(trial(model,forecaster,terrain,seed))
    values=report(rows)
    path=ROOT/'survey_selection.json'
    if stage=='validation':
        values['comparison']=min(METHODS[:-1],key=lambda name:values['pooled'][name])
    else:
        comparator=json.loads(path.read_text())['comparison']
        values['comparison']=comparator
        baseline=values['pooled'][comparator]
        values['relativeGain']=1-values['pooled']['targeted surveys']/baseline if baseline else 0
    (path if stage=='validation' else ROOT/'survey_holdout.json').write_text(json.dumps(values,indent=2)+'\n')
    print(json.dumps(values,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('validation','test'))
    main(parser.parse_args().stage)
