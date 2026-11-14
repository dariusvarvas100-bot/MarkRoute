"""Attach unchanged terrain features and evaluate the remote-region split."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
LOCAL_REPLAY = not (HERE / 'map_model.py').exists()
ARCHIVE = HERE.parent if LOCAL_REPLAY else HERE
ROOT = ARCHIVE.parent.parent / 'outputs' / 'MarkRoute' if LOCAL_REPLAY else HERE
REMOTE_PANEL = HERE / ('panel.csv' if LOCAL_REPLAY else 'remote_panel.csv')
REMOTE_RESULTS = HERE / ('results.json' if LOCAL_REPLAY else 'remote_results.json')
REMOTE_KRIGING = HERE / ('kriging_results.json' if LOCAL_REPLAY else 'remote_kriging_results.json')
sys.path.insert(0, str(ROOT))

from terrain import FEET_PER_METRE, ZOOM, web_coordinate


def tile(key):
    x,y=key
    name=f'{x}_{y}.png'
    location=HERE/'terrain_cache'/name
    candidates=[location, ARCHIVE/'terrain_cache'/name, ARCHIVE/'external'/'terrain_cache'/name]
    source=f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{ZOOM}/{x}/{y}.png'
    try:
        for path in candidates:
            if path.is_file():
                raw=path.read_bytes()
                break
        else:
            raw=subprocess.check_output(['curl','-fLsS','--max-time','18','--retry','2',source],stderr=subprocess.PIPE)
        location.write_bytes(raw)
        with Image.open(location) as picture:
            red,green,blue=np.asarray(picture.convert('RGB'),dtype=np.float32).transpose(2,0,1)
        return key,(red*256+green+blue/256-32768)*FEET_PER_METRE,sha256(raw).hexdigest()
    except (subprocess.CalledProcessError,OSError,ValueError):
        return key,None,None


def prepare_terrain():
    panel=pd.read_csv(HERE/'filtered_marks.csv',keep_default_na=False)
    x,y=web_coordinate(panel.latitude.to_numpy(),panel.longitude.to_numpy())
    keys=list(zip(x.astype(int)//256,y.astype(int)//256))
    (HERE/'terrain_cache').mkdir(exist_ok=True)
    print(f'{len(panel)} eligible marks, {len(set(keys))} unique tiles',flush=True)
    with ThreadPoolExecutor(max_workers=8) as workers:
        rasters={key:(pixels,digest) for key,pixels,digest in workers.map(tile,sorted(set(keys)))}
    prepared=[]
    for key,cx,cy in zip(keys,x,y):
        raster,_=rasters[key]
        if raster is None:
            prepared.append((np.nan,np.nan,np.nan))
            continue
        ix,iy=int(cx)%256,int(cy)%256
        sample=raster[max(0,iy-3):iy+4,max(0,ix-3):ix+4]
        relief=raster[max(0,iy-8):iy+9,max(0,ix-8):ix+9]
        prepared.append((raster[iy,ix],float(np.median(sample)),float(np.percentile(relief,90)-np.percentile(relief,10))))
    panel[['terrain_ft','terrain_median_ft','relief_ft']]=prepared
    missing_events=set(panel.loc[panel.terrain_ft.isna(),'event_id'])
    panel=panel.loc[~panel.event_id.isin(missing_events)]
    eligibility=json.loads((HERE/'eligibility.json').read_text())
    for event in eligibility:
        if event['id'] in missing_events:
            event['excluded']='at least one source raster unavailable'
    locked={'eligibility':eligibility,'terrainSha256':{f'{x}/{y}':digest for (x,y),(_,digest) in rasters.items()}}
    (HERE/'manifest.json').write_text(json.dumps(locked,indent=2)+'\n')
    panel.to_csv(REMOTE_PANEL,index=False)
    print(panel.groupby('event_name').size().to_string(),flush=True)


def evaluate():
    import torch
    from threadpoolctl import threadpool_limits
    from map_model import TRAIN, STATIC_NAMES, fit_boosted, sample, bootstrap
    torch.set_num_threads(4)
    points=pd.read_csv(ROOT/'panel.csv',keep_default_na=False)
    extra=pd.read_csv(REMOTE_PANEL,keep_default_na=False)
    assert not set(points.event_id)&set(extra.event_id)
    baseline='bounded local regression'
    model_name='gradient boost'
    if extra.empty:
        raise SystemExit('No qualifying remote events have complete terrain.')
    with threadpool_limits(4):
        model=fit_boosted(sample(points,TRAIN,18))
        replay=sample(extra,extra.event_id.unique(),16)
        estimates={baseline:replay['baseline'][:,STATIC_NAMES.index(baseline)],model_name:model.predict(replay['features'])+replay['default']}
    actual=replay['actual']
    error={method:np.abs(values-actual) for method,values in estimates.items()}
    far_keys={(row.event_name,row.site_id) for row in extra.loc[extra.distance_to_prior_km.gt(100)].itertuples()}
    only_far=np.array([(event,site) in far_keys for event,site in zip(replay['event'],replay['site'])])

    def rates(keep):
        values={method:{'maeFt':float(part[keep].mean()),'gt5Ft':int((part[keep]>5).sum()),
                        'gt5FtRate':float((part[keep]>5).mean()),'p99Ft':float(np.quantile(part[keep],.99))}
                for method,part in error.items()}
        return {'maskSitePairs':int(keep.sum()), **values,
                'relativeMaeReduction':1-values[model_name]['maeFt']/values[baseline]['maeFt']}

    def summarize(qualifies):
        intervals=bootstrap({key:replay[key][qualifies] for key in ('actual','event','block')},estimates[model_name][qualifies],estimates[baseline][qualifies])
        named={event:rates(qualifies&(replay['event']==event)) for event in np.unique(replay['event'][qualifies])}
        return {'pooled':rates(qualifies),'events':named,'conditionalPairedSpatial95Ft':intervals}

    by_count=extra.groupby('event_name').agg(sites=('site_id','size'),sitesFar=('distance_to_prior_km',lambda x:int(x.gt(100).sum())),
                                          medianKm=('distance_to_prior_km','median'),maxKm=('distance_to_prior_km','max'))
    output={'modelSha256':sha256((ROOT/'map_model.py').read_bytes()).hexdigest(),'sitesAndDistances':by_count.to_dict(orient='index'),
            'all':summarize(np.ones(len(actual),dtype=bool)),'beyond100Km':summarize(only_far)}
    main=output['all']
    output['passed']=bool(main['pooled']['relativeMaeReduction']>=.10 and main['conditionalPairedSpatial95Ft'][0]>0 and
                          min(e['relativeMaeReduction'] for e in main['events'].values())>=-.05)
    REMOTE_RESULTS.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2),flush=True)


def evaluate_kriging():
    from threadpoolctl import threadpool_limits

    from kriging import BASELINE, ESTIMATOR, KRIGING, SETTINGS, collect, source_hash, summarize, synthetic_check
    from map_model import STATIC_NAMES, TRAIN, fit_boosted, sample

    synthetic_check()
    lock = json.loads((ROOT / 'kriging_selection.json').read_text())
    assert lock['originalPanelSha256'] == source_hash(ROOT / 'panel.csv')
    assert lock['scriptSha256'] == source_hash(ROOT / 'kriging.py')
    assert lock['modelFileSha256'] == source_hash(ROOT / 'map_model.py')
    selected = tuple(lock['selected'])
    assert selected in SETTINGS
    points = pd.read_csv(ROOT / 'panel.csv', keep_default_na=False)
    remote = pd.read_csv(REMOTE_PANEL, keep_default_na=False)
    storms = remote.event_id.unique()
    with threadpool_limits(4):
        model = fit_boosted(sample(points, TRAIN, 18))
        data = sample(remote, storms, 16)
        estimates = {BASELINE: data['baseline'][:, STATIC_NAMES.index(BASELINE)],
                     ESTIMATOR: model.predict(data['features']) + data['default'],
                     KRIGING: collect(remote, storms, 16, (selected,), check=True)[selected]}
    report = summarize(data, estimates)
    expected = json.loads(REMOTE_RESULTS.read_text())['all']
    assert report['predictivePairs'] == expected['pooled']['maskSitePairs']
    for storm, scores in report['events'].items():
        for method in (BASELINE, ESTIMATOR):
            np.testing.assert_allclose(scores[method]['maeFt'], expected['events'][storm][method]['maeFt'], rtol=0, atol=1e-12)
    output = {'selectedSetting': selected, 'primaryResultSha256': source_hash(REMOTE_RESULTS), **report}
    encoded = json.dumps(output, indent=2, allow_nan=False) + '\n'
    REMOTE_KRIGING.write_text(encoded)
    print(encoded, end='', flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('terrain','evaluate','kriging'), nargs='?', default='evaluate')
    stages = {'terrain': prepare_terrain, 'evaluate': evaluate, 'kriging': evaluate_kriging}
    stages[parser.parse_args().stage]()
