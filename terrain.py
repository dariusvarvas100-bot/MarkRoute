"""Prepare USGS marks and sample independent Mapzen ground elevations."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import json
import math
import subprocess

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
TRAIN=(24,135,180,182)
DEV=(283,287)
TEST=(325,346)
ZOOM=12
FEET_PER_METRE=3.280839895
TILES=ROOT/'terrain_cache'


def locations():
    panels=[]
    for event in (*TRAIN,*DEV,*TEST):
        data=pd.read_json(ROOT/f'event{event}.json')
        scoped=data[data.vertical_datum_name.eq('NAVD88') & data.hwm_quality_id.isin([1,2,3]) &
                    data.environment.isin(['Coastal','Riverine']) &
                    data[['latitude','longitude','elev_ft','site_id']].notna().all(axis=1)].copy()
        scoped=scoped.sort_values(['hwm_quality_id','id']).drop_duplicates('site_id')
        scoped['loncell']=np.floor(scoped.longitude*600).astype(int)
        scoped['latcell']=np.floor(scoped.latitude*600).astype(int)
        scoped=scoped.drop_duplicates(['loncell','latcell'])
        sample=scoped[['event_id','event_name','site_id','latitude','longitude','elev_ft','waterbody','environment']].copy()
        sample['waterbody']=sample.waterbody.fillna('').str.casefold().str.strip().str.replace(r'[^\w]+',' ',regex=True)
        sample.loc[sample.waterbody.isin(['unknown','n a','na','unnamed']),'waterbody']=''
        panels.append(sample)
    return pd.concat(panels, ignore_index=True)


def web_coordinate(lat,lon):
    dimension=2**ZOOM*256
    x=(lon+180)/360*dimension
    y=(1-np.arcsinh(np.tan(np.deg2rad(lat)))/np.pi)/2*dimension
    return x,y


def tile(key):
    tx,ty=key
    dest=TILES/f'{tx}_{ty}.png'
    uri=f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{ZOOM}/{tx}/{ty}.png'
    if not dest.is_file():
        tmp=dest.with_suffix('.partial')
        subprocess.run(['curl','-fLsS','--retry','2','--max-time','22',uri,'-o',str(tmp)],check=True)
        tmp.replace(dest)
    with Image.open(dest) as image:
        red,green,blue=np.asarray(image.convert('RGB'),dtype=np.float32).transpose(2,0,1)
    return key,(red*256+green+blue/256-32768)*FEET_PER_METRE


def main():
    TILES.mkdir(exist_ok=True)
    frame=locations()
    x,y=web_coordinate(frame.latitude.to_numpy(),frame.longitude.to_numpy())
    keys=list(zip(x.astype(int)//256,y.astype(int)//256))
    unique=set(keys)
    # Three-pixel neighborhoods never cross into a tile without explicitly fetching it.
    print(f'{len(frame)} retained marks and {len(unique)} source tiles',flush=True)
    with ThreadPoolExecutor(max_workers=10) as executor:
        cache=dict(executor.map(tile,sorted(unique)))
    features=[]
    for key,cx,cy in zip(keys,x,y):
        raster=cache[key]
        ix,iy=int(cx)%256,int(cy)%256
        p=raster[iy,ix]
        slope=raster[max(0,iy-3):iy+4,max(0,ix-3):ix+4]
        terrain=raster[max(0,iy-8):iy+9,max(0,ix-8):ix+9]
        features.append((p,float(np.median(slope)),float(np.percentile(terrain,90)-np.percentile(terrain,10))))
    frame[['terrain_ft','terrain_median_ft','relief_ft']]=features
    frame.to_csv(ROOT/'panel.csv',index=False)
    print(frame.groupby(['event_name','environment']).size().to_string(),flush=True)


if __name__=='__main__':
    main()
