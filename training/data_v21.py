"""V2.1 固定场景划分、特征审计和数据集。"""
import csv,json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset
LABELS=['Safe','Warning','Emergency'];LABEL_MAP={v:i for i,v in enumerate(LABELS)}
ALL=['ego_speed','lead_speed','distance','relative_speed','ttc','ego_acceleration','road_friction','road_slope','vehicle_mass','reaction_time','sensor_delay_ms','brake_state']
FEATURE_SETS={'without_brake_state':[x for x in ALL if x!='brake_state'],'without_ego_acceleration':[x for x in ALL if x!='ego_acceleration'],'without_both':[x for x in ALL if x not in {'brake_state','ego_acceleration'}]}
AUDIT_NAMES={'without_brake_state':'primary','without_ego_acceleration':'without_ego_acceleration','without_both':'without_both'}
def read(path):
 with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def _scores(train,validation,features):
 out={}
 for f in features:
  x=np.asarray([float(r[f]) for r in train]);y=np.asarray([LABEL_MAP[r['risk_level']] for r in train]);edges=np.unique(np.quantile(x,np.linspace(0,1,9)));b=np.clip(np.digitize(x,edges[1:-1]),0,7);majority=[np.bincount(y[b==i],minlength=3).argmax() if np.any(b==i) else 0 for i in range(8)];xv=np.asarray([float(r[f]) for r in validation]);yv=np.asarray([LABEL_MAP[r['risk_level']] for r in validation]);bv=np.clip(np.digitize(xv,edges[1:-1]),0,7);out[f]=round(float((np.asarray([majority[i] for i in bv])==yv).mean()),4)
 return out
def audit(feature_set='without_brake_state'):
 parts={n:read(f'data/v21_splits/{n}.csv') for n in ['train','validation','test']};groups={n:{r['scenario_id'] for r in rows} for n,rows in parts.items()};overlap=(groups['train']&groups['validation'])|(groups['train']&groups['test'])|(groups['validation']&groups['test']);features=FEATURE_SETS[feature_set];forbidden={'scenario_id','risk_level','stopping_distance'};errors=[];delay_mismatch=[]
 for r in parts['train']+parts['validation']+parts['test']:
  rel,dist,ttc=float(r['relative_speed']),float(r['distance']),float(r['ttc']);expected=dist/rel if rel>.1 else 99.;errors.append(abs(ttc-expected)/max(1.,expected));delay_mismatch.append(int(r['sensor_delay_ms'])!=int(r['effective_delay_ms']))
 if overlap or set(features)&forbidden or max(errors)>.05 or any(delay_mismatch):raise RuntimeError(f'V2.1 audit failed: overlap={len(overlap)}, ttc={max(errors):.4f}, delay_mismatch={sum(delay_mismatch)}')
 scores=_scores(parts['train'],parts['validation'],ALL);summary={'scenario_counts':{n:len(g) for n,g in groups.items()},'scenario_overlap':len(overlap),'feature_set':feature_set,'features':features,'forbidden_features_found':sorted(set(features)&forbidden),'ttc_max_relative_recompute_error':round(max(errors),5),'sensor_delay_matches_effective_delay':True,'effective_delay_ms_values':sorted({int(r['effective_delay_ms']) for r in parts['train']+parts['validation']+parts['test']}),'scaler_fit':'train only','single_feature_validation_accuracy':scores,'suspicious_single_features_over_0_95':[k for k,v in scores.items() if v>.95]};root=Path('results/v21');root.mkdir(parents=True,exist_ok=True);(root/f"data_audit_{AUDIT_NAMES[feature_set]}.json").write_text(json.dumps(summary,indent=2),encoding='utf-8');return parts,summary
def fit_scaler(rows,features):
 x=np.asarray([[float(r[f]) for f in features] for r in rows],dtype=np.float32);return x.mean(0),x.std(0).clip(1e-6)
def arrays(rows,mean,std,features):
 x=(np.asarray([[float(r[f]) for f in features] for r in rows],dtype=np.float32)-mean)/std;return x,np.asarray([LABEL_MAP[r['risk_level']] for r in rows],dtype=np.int64),np.asarray([float(r['stopping_distance']) for r in rows],dtype=np.float32)
class Tabular(Dataset):
 def __init__(self,rows,mean,std,features):self.x,self.y,self.d=arrays(rows,mean,std,features)
 def __len__(self):return len(self.y)
 def __getitem__(self,i):return torch.from_numpy(self.x[i]),torch.tensor(self.y[i]),torch.tensor(self.d[i])
class Sequences(Dataset):
 def __init__(self,rows,mean,std,features,length=10):
  by={};self.items=[]
  for r in rows:by.setdefault(r['scenario_id'],[]).append(r)
  for scenario in by.values():
   x,y,d=arrays(scenario,mean,std,features)
   for i in range(length-1,len(scenario)):self.items.append((x[i-length+1:i+1],y[i],d[i]))
 def __len__(self):return len(self.items)
 def __getitem__(self,i):x,y,d=self.items[i];return torch.from_numpy(x),torch.tensor(y),torch.tensor(d)
def endpoint_rows(rows,length=10):
 by={};out=[]
 for r in rows:by.setdefault(r['scenario_id'],[]).append(r)
 for scenario in by.values():out.extend(scenario[length-1:])
 return out
