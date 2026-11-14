"""Predict the water-surface elevation at flagged sites from partial field surveys."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
from torch import nn

from terrain import DEV, TEST, TRAIN

ROOT=Path(__file__).resolve().parent
OBSERVED=.15
PEERS=12
SEED=7162
EPOCHS=(2,5,10,16,24,32)
STATIC_NAMES=('nearest', 'geographic interpolation', 'terrain plus local depth', 'bounded local regression')


def xy_km(points):
    lat=points.latitude.to_numpy()
    lon=points.longitude.to_numpy()
    return np.c_[111.2*np.cos(np.deg2rad(lat.mean()))*(lon-lon.mean()),111.2*(lat-lat.mean())]


def observation_features(frame, xy, distances, seen, queries):
    ranking=np.argsort(distances[np.ix_(queries,seen)],axis=1)
    count=min(PEERS,len(seen))
    neighbors=seen[ranking[:,:count]]
    mask=np.ones(neighbors.shape,bool)
    if count<PEERS:
        gaps=np.broadcast_to(neighbors[:,0,None],(len(neighbors),PEERS-count))
        neighbors=np.c_[neighbors,gaps]
        mask=np.c_[mask,np.zeros_like(gaps,dtype=bool)]
    ground=frame.terrain_median_ft.to_numpy()
    point=frame.terrain_ft.to_numpy()
    relief=frame.relief_ft.to_numpy()
    water=frame.waterbody.to_numpy()
    y=frame.elev_ft.to_numpy()
    coastal=frame.environment.eq('Coastal').to_numpy()
    d=distances[queries[:,None],neighbors]
    displacement=xy[queries,None,:]-xy[neighbors]
    same=(water[queries,None]==water[neighbors]) & (water[queries,None]!='')
    dg=ground[queries,None]-ground[neighbors]
    surface=y[neighbors]
    corrected=surface+dg
    regional=np.broadcast_to(coastal[queries,None],d.shape)
    local_depth=surface-ground[neighbors]
    pair=np.stack([np.log1p(d), np.arcsinh(dg/15), np.arcsinh(local_depth/10), same,regional,
                   np.log1p(relief[queries,None]+np.zeros_like(d)),np.log1p(relief[neighbors]),
                   np.arcsinh((point-ground)[queries,None]/5)+np.zeros_like(d),
                   np.arcsinh((point-ground)[neighbors]/5),
                   np.arcsinh(displacement[:,:,0]/5),np.arcsinh(displacement[:,:,1]/5)],axis=-1)
    w=mask/(.25+d)**1.5
    w/=w.sum(axis=1,keepdims=True)
    avg=(w*surface).sum(1)
    dz=(w*dg).sum(1)
    terrain=ground[queries]+np.nanmedian(np.where(mask,local_depth,np.nan),axis=1)
    regression=((w*(surface-avg[:,None])*(-dg+dz[:,None])).sum(1)/
                ((w*(dg-dz[:,None])**2).sum(1)+4))
    bounded=avg+np.clip(regression,0,1.2)*dz
    baselines=np.c_[surface[:,0],avg,terrain,bounded]
    default=np.where(coastal[queries],avg,terrain)
    features=np.c_[pair[:,0],pair[:,1],np.einsum('ij,ijk->ik',w,pair),
                   np.arcsinh((baselines-default[:,None])/8),np.log1p(mask.sum(axis=1)),
                   np.arcsinh(ground[queries]/50)]
    return {'pair':pair.astype('float32'),'mask':mask,'values':np.stack([surface,corrected],-1).astype('float32'),
            'baseline':baselines,'features':np.rint(features*1e6).astype('int64'),'default':default,'actual':y[queries]}


def split_points(frame, seed):
    rng=np.random.default_rng(seed)
    seen=[]
    for idx in frame.groupby('environment').indices.values():
        qty=min(len(idx)-1,max(2,round(len(idx)*OBSERVED)))
        seen.extend(rng.choice(idx,qty,replace=False))
    seen=np.sort(seen)
    queries=np.setdiff1d(np.arange(len(frame)),seen)
    return seen,queries


def episode(frame, seed):
    frame=frame.reset_index(drop=True)
    xy=xy_km(frame)
    d=cdist(xy,xy)
    seen,queries=split_points(frame,seed)
    result={}
    for environment,ids in frame.groupby('environment').indices.items():
        index=np.intersect1d(queries,ids)
        if not len(index):
            continue
        context=np.intersect1d(seen,ids)
        if not len(context):
            context=seen
        data=observation_features(frame,xy,d,context,index)
        data['event']=frame.event_name.to_numpy()[index]
        data['environment']=frame.environment.to_numpy()[index]
        data['site']=frame.site_id.to_numpy()[index]
        data['block']=np.asarray([f'{int(x//10)}_{int(y//10)}' for x,y in xy[index]])
        for key,value in data.items():
            result.setdefault(key,[]).append(value)
    return {key:np.concatenate(value) for key,value in result.items()}


def sample(panel, storms, number):
    result={}
    for event in storms:
        data=panel[panel.event_id==event]
        for seed in range(SEED,SEED+number):
            for key,value in episode(data,seed).items():
                result.setdefault(key,[]).append(value)
    return {key:np.concatenate(value) for key,value in result.items()}


class WaterAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.context=nn.Sequential(nn.Linear(11,48),nn.LayerNorm(48),nn.GELU(),
                                   nn.Linear(48,48),nn.GELU(),nn.Linear(48,3))

    def forward(self, pair, mask, values):
        learned=self.context(pair)
        logits=learned[:,:,:2]-1.5*pair[:,:,:1]
        # A low relief coast can transfer the measured surface. A steep river
        # often transfers height above nearby ground better than absolute height.
        logits=logits+torch.stack([pair[:,:,4],1-pair[:,:,4]],-1)*2
        logits=logits.masked_fill(~mask[:,:,None],-torch.inf)
        w=logits.flatten(1).softmax(1).reshape_as(logits)
        surface=(w*values).sum((1,2))
        adjustment=(w.sum(-1)*learned[:,:,2].tanh()).sum(1)*6
        return surface+adjustment


def tensors(data):
    return [torch.from_numpy(data[name]) for name in ('pair','mask','values')]


def scores(model, data):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(*args) for args in zip(*(part.split(1024) for part in tensors(data)))]).numpy()


def mae(actual,predicted):
    return float(np.abs(actual-predicted).mean())


def train_neural(train,dev):
    torch.manual_seed(SEED)
    rng=torch.Generator().manual_seed(SEED+1)
    model=WaterAttention()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.04)
    features=tensors(train)
    target=torch.tensor(train['actual'],dtype=torch.float32)
    criterion=nn.SmoothL1Loss(beta=2)
    best=(float('inf'),0)
    for epoch in range(1,max(EPOCHS)+1):
        model.train()
        for idx in torch.randperm(len(target),generator=rng).split(384):
            optimizer.zero_grad()
            loss=criterion(model(*(part[idx] for part in features)),target[idx])
            loss.backward()
            optimizer.step()
        if epoch in EPOCHS:
            error=mae(dev['actual'],scores(model,dev))
            print(epoch,round(error,4),flush=True)
            if error<best[0]:
                best=error,epoch
                selected=deepcopy(model.state_dict())
    model.load_state_dict(selected)
    return model,best


def all_predictions(model,boosted,data):
    return {**dict(zip(STATIC_NAMES,data['baseline'].T)),
            'gradient boost': boosted.predict(data['features'])+data['default'],
            'terrain attention': scores(model,data)}


def metrics(predictions,data):
    actual=data['actual']
    def summarize(which):
        return {method:{'maeFt':mae(actual[which],p[which]),
                       'missOver5FtRate':float((np.abs(actual[which]-p[which])>5).mean())}
                for method,p in predictions.items()}
    return {'pooled':summarize(np.ones(len(actual),bool)),
            'events':{e:summarize(data['event']==e) for e in np.unique(data['event'])},
            'environments':{e:summarize(data['environment']==e) for e in np.unique(data['environment'])}}


def bootstrap(data,prediction,baseline):
    diff=np.abs(data['actual']-baseline)-np.abs(data['actual']-prediction)
    rng=np.random.default_rng(SEED)
    folds=[]
    for event in np.unique(data['event']):
        keep=data['event']==event
        _,idx=np.unique(data['block'][keep],return_inverse=True)
        gain=np.bincount(idx,weights=diff[keep])
        count=np.bincount(idx)
        draw=rng.integers(len(count),size=(4200,len(count)))
        folds.append((gain[draw].sum(1),count[draw].sum(1)))
    numerator,denominator=np.asarray(folds).sum(0)
    return np.quantile(numerator/denominator,[.025,.975]).tolist()


def fit_boosted(data):
    model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.08,max_iter=180,
                                        max_leaf_nodes=18,l2_regularization=1.,random_state=SEED)
    return model.fit(data['features'],data['actual']-data['default'])


def run(stage):
    torch.set_num_threads(4)
    panel=pd.read_csv(ROOT/'panel.csv',keep_default_na=False)
    train=sample(panel,TRAIN,18)
    dev=sample(panel,DEV,10)
    boosted=fit_boosted(train)
    selection=ROOT/'selection.json'
    checkpoint=ROOT/'water_attention.pth'
    if stage=='validation':
        model,best=train_neural(train,dev)
        values=metrics(all_predictions(model,boosted,dev),dev)
        comparator=min((k for k in values['pooled'] if k!='terrain attention'),key=lambda k:values['pooled'][k]['maeFt'])
        selection.write_text(json.dumps({'comparator':comparator,'chosenEpoch':best[1],'validation':values},indent=2)+'\n')
        torch.save(model.state_dict(),checkpoint)
        print(json.dumps(values,indent=2),flush=True)
        return
    frozen=json.loads(selection.read_text())
    model=WaterAttention()
    model.load_state_dict(torch.load(checkpoint,weights_only=True))
    holdout=sample(panel,TEST,16)
    predictions=all_predictions(model,boosted,holdout)
    values=metrics(predictions,holdout)
    name=frozen['comparator']
    interval=bootstrap(holdout,predictions['terrain attention'],predictions[name])
    def gain(row):
        return 1-row['terrain attention']['maeFt']/row[name]['maeFt']
    relative=gain(values['pooled'])
    by_event={k:gain(v) for k,v in values['events'].items()}
    gate={'comparator':name,'relativeErrorReduction':relative,'pairedSpatial95IntervalFt':interval,
          'eventReductions':by_event,
          'passed':bool(relative>=.1 and interval[0]>0 and min(by_event.values())>=-.05)}
    validation=frozen['validation']['pooled']
    operational=min(validation,key=lambda method:validation[method]['maeFt'])
    simpler=min(STATIC_NAMES,key=lambda method:validation[method]['maeFt'])
    local_gains={key:1-row[operational]['maeFt']/row[simpler]['maeFt']
                 for key,row in values['events'].items()}
    practical={'frozenWinner':operational,'frozenHeuristic':simpler,'eventRelativeGains':local_gains,
               'relativeErrorReduction':1-values['pooled'][operational]['maeFt']/values['pooled'][simpler]['maeFt'],
               'pairedSpatial95IntervalFt':bootstrap(holdout,predictions[operational],predictions[simpler])}
    practical['passed']=bool(practical['relativeErrorReduction']>=.10 and
                             practical['pairedSpatial95IntervalFt'][0]>0 and min(local_gains.values())>=-.05)
    report={'nLocations':len(np.unique(holdout['site'])),'evaluatedMaskLocationPairs':len(holdout['site']),
            'results':values,'gate':gate,'practicalComparison':practical}
    (ROOT/'holdout.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('validation','test'))
    run(parser.parse_args().stage)
