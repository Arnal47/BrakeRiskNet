import csv,json,random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

# 只含当前可观测状态；不含 scenario_id、risk_level、stopping_distance 或任何未来/隐藏参数。
FEATURES=['ego_speed','lead_speed','distance','relative_speed','ttc','ego_acceleration','road_friction','road_slope','vehicle_mass','reaction_time','sensor_delay_ms','brake_state']
LABEL_MAP={'Safe':0,'Warning':1,'Emergency':2}; LABELS=['Safe','Warning','Emergency']
def read(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def audit():
    parts={name:read(f'data/v15_splits/{name}.csv') for name in ['train','validation','test']}; groups={k:{r['scenario_id'] for r in v} for k,v in parts.items()}; overlap=(groups['train']&groups['validation'])|(groups['train']&groups['test'])|(groups['validation']&groups['test'])
    forbidden={'scenario_id','risk_level','stopping_distance'}; bad=set(FEATURES)&forbidden
    if overlap or bad: raise RuntimeError(f'Leakage audit failed: overlap={overlap}, forbidden={bad}')
    summary={'scenario_counts':{k:len(v) for k,v in groups.items()},'scenario_overlap':len(overlap),'features':FEATURES,'forbidden_features_found':sorted(bad),'scaler_fit':'train only'}; Path('results/v2').mkdir(parents=True,exist_ok=True); Path('results/v2/data_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); return parts,summary
def fit_scaler(rows):
    x=np.array([[float(r[f]) for f in FEATURES] for r in rows],dtype=np.float32); return x.mean(0),x.std(0).clip(1e-6)
def arrays(rows,mean,std):
    x=(np.array([[float(r[f]) for f in FEATURES] for r in rows],dtype=np.float32)-mean)/std; y=np.array([LABEL_MAP[r['risk_level']] for r in rows],dtype=np.int64); d=np.array([float(r['stopping_distance']) for r in rows],dtype=np.float32); return x,y,d
class Tabular(Dataset):
    def __init__(self,rows,mean,std): self.x,self.y,self.d=arrays(rows,mean,std)
    def __len__(self):return len(self.y)
    def __getitem__(self,i):return torch.from_numpy(self.x[i]),torch.tensor(self.y[i]),torch.tensor(self.d[i])
class Sequences(Dataset):
    def __init__(self,rows,mean,std,length=10):
        by={};
        for r in rows: by.setdefault(r['scenario_id'],[]).append(r)
        self.items=[]
        for scenario in by.values():
            x,y,d=arrays(scenario,mean,std)
            for i in range(length-1,len(scenario)): self.items.append((x[i-length+1:i+1],y[i],d[i]))
    def __len__(self):return len(self.items)
    def __getitem__(self,i): x,y,d=self.items[i]; return torch.from_numpy(x),torch.tensor(y),torch.tensor(d)
