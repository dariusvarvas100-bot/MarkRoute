"""Evaluate flags for large errors in the frozen MarkRoute elevation model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingClassifier

from map_model import DEV, ROOT, SEED, TEST, TRAIN, bootstrap, fit_boosted
from map_model import observation_features, sample, split_points, xy_km


BASELINES = ('nearest survey distance', 'terrain relief', 'model disagreement')
SIGNALS = (*BASELINES, 'local recheck')
BUDGET=.20
THRESHOLD_FT=5.
REPLAYS=16


def local_rechecks(frame,xy,distances,context,queries,model):
    if len(context)<2:
        return np.zeros(len(queries))
    # Infer a labeled site's error using only the other revealed surveys.
    excluded=observation_features(frame,xy,distances,context,context,exclude_self=True)
    estimates=excluded['default']+model.predict(excluded['features'])
    residual=np.abs(excluded['actual']-estimates)
    ground=frame.terrain_median_ft.to_numpy()
    water=frame.waterbody.to_numpy()
    distance=distances[np.ix_(queries,context)]
    relief_gap=np.abs(ground[queries,None]-ground[context])
    same_water=(water[queries,None]==water[context]) & (water[queries,None]!='')
    weight=(1.+same_water)/(0.25+distance)**1.5/(1.+relief_gap/30.)
    return weight@residual/weight.sum(axis=1)


def episode(frame,model,seed):
    frame=frame.reset_index(drop=True)
    xy=xy_km(frame)
    distances=cdist(xy,xy)
    seen,queries=split_points(frame,seed)
    records=[]
    for ids in frame.groupby('environment').indices.values():
        wanted=np.intersect1d(queries,ids)
        context=np.intersect1d(seen,ids)
        if not len(wanted):
            continue
        if not len(context):
            context=seen
        features=observation_features(frame,xy,distances,context,wanted)
        predictions=features['default']+model.predict(features['features'])
        disagreement=np.abs(predictions-features['baseline'][:,3])
        check=local_rechecks(frame,xy,distances,context,wanted,model)
        distance=distances[np.ix_(wanted,context)].min(axis=1)
        relief=frame.relief_ft.to_numpy()[wanted]
        score_inputs=np.c_[features['features'],np.rint(np.c_[disagreement,check]*1e6)]
        absolute_error=np.abs(features['actual']-predictions)
        records.append({'x':score_inputs,'y':absolute_error>THRESHOLD_FT,
                        'error':absolute_error,'seed':np.full(len(wanted),seed),
                        'event':frame.event_name.to_numpy()[wanted],
                        'site':frame.site_id.to_numpy()[wanted],
                        'block':np.asarray([f'{int(x//10)}_{int(y//10)}' for x,y in xy[wanted]]),
                        **dict(zip(SIGNALS,(distance,relief,disagreement,check)))})
    return {key:np.concatenate([item[key] for item in records]) for key in records[0]}


def stack(records):
    return {key:np.concatenate([item[key] for item in records]) for key in records[0]}


def replay(panel,storms,model,count=REPLAYS):
    return stack([episode(panel[panel.event_id.eq(event)],model,seed)
                  for event in storms for seed in range(SEED,SEED+count)])


def choose(scores,data):
    flags=np.zeros(len(scores),dtype=bool)
    for event in np.unique(data['event']):
        for seed in np.unique(data['seed']):
            index=np.flatnonzero((data['event']==event)&(data['seed']==seed))
            take=int(np.floor(BUDGET*len(index)))
            ordering=np.lexsort((data['site'][index],-scores[index]))
            flags[index[ordering[:take]]]=True
    return flags


def assess(flags,data):
    risky=data['y']
    def measure(keep):
        selected=flags&keep
        left=keep&~flags
        hits=int(np.sum(risky&selected))
        risky_total=int(np.sum(risky&keep))
        return {'withheldPairs':int(np.sum(keep)),
                'reviewedPairs':int(np.sum(selected)),
                'largeErrorPairs':risky_total,
                'largeErrorsFlagged':hits,
                'largeErrorRecall':hits/risky_total if risky_total else None,
                'unflaggedLargeErrorRate':float(np.mean(risky[left])) if np.any(left) else None}
    return {'pooled':measure(np.ones(len(risky),bool)),
            'events':{event:measure(data['event']==event) for event in np.unique(data['event'])}}


def paired_interval(candidate,reference,data):
    # Resample repetitions jointly within each ten-kilometre block and storm.
    # The measured policy queues remain fixed while resampling.
    delta=(candidate.astype(int)-reference.astype(int))*data['y']
    rng=np.random.default_rng(SEED)
    numerators=[]
    denominators=[]
    for event in np.unique(data['event']):
        keep=data['event']==event
        _,indices=np.unique(data['block'][keep],return_inverse=True)
        captured_delta=np.bincount(indices,weights=delta[keep])
        errors=np.bincount(indices,weights=data['y'][keep])
        draws=rng.integers(len(errors),size=(4200,len(errors)))
        numerators.append(captured_delta[draws].sum(axis=1))
        denominators.append(errors[draws].sum(axis=1))
    count=np.sum(denominators,axis=0)
    diff=np.sum(numerators,axis=0)
    return np.percentile(diff[count>0]/count[count>0]*100,[2.5,97.5]).tolist()


def evaluate(panel,external):
    # Each training storm is withheld from its own elevation-model fit when
    # constructing the labels for the large-error ranker.
    training=[]
    for held in TRAIN:
        model=fit_boosted(sample(panel,[event for event in TRAIN if event!=held],18))
        training.append(replay(panel,[held],model,12))
    training=stack(training)
    ranker=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=9,min_samples_leaf=80,
                                          l2_regularization=3.,learning_rate=.07,random_state=SEED)
    ranker.fit(training['x'],training['y'])
    model=fit_boosted(sample(panel,TRAIN,18))
    cohorts={
        'development':replay(panel,DEV,model),
        'primary':replay(panel,TEST,model),
        'external':replay(external,external.event_id.unique(),model),
    }
    scores={}
    flags={}
    for key,data in cohorts.items():
        scores[key]={name:data[name] for name in SIGNALS}
        scores[key]['learned review ranker']=ranker.predict_proba(data['x'])[:,1]
        flags[key]={name:choose(raw,data) for name,raw in scores[key].items()}
    dev=cohorts['development']
    comparator=max(BASELINES,key=lambda name:np.sum(flags['development'][name]&dev['y']))
    methods=('learned review ranker','local recheck')
    results={}
    for key,data in {'development':dev,'primary':cohorts['primary'],'external':cohorts['external'],
                    'all test':stack([cohorts['primary'],cohorts['external']])}.items():
        current=(flags[key] if key in flags else
                 {name:np.r_[flags['primary'][name],flags['external'][name]] for name in flags['primary']})
        results[key]={'methods':{name:assess(mask,data) for name,mask in current.items()},
                      'pairedRecallGainPoints':{name:paired_interval(current[name],current[comparator],data)
                                                for name in methods}}
    winner=max(methods,key=lambda name:results['development']['methods'][name]['pooled']['largeErrorsFlagged'])
    return {'budgetFraction':BUDGET,'thresholdFt':THRESHOLD_FT,'developmentBestBaseline':comparator,
            'developmentSelectedNewMethod':winner,'results':results},ranker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'review_results.json')
    args=parser.parse_args()
    panel=pd.read_csv(ROOT/'panel.csv',keep_default_na=False)
    external=pd.read_csv(ROOT/'external_panel.csv',keep_default_na=False)
    result,_=evaluate(panel,external)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
