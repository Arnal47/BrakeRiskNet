"""仅评价固定 V2.1 test split，并保存可对齐的 endpoint 元数据。"""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from training.data_v21 import audit,Tabular,Sequences,LABELS,endpoint_rows
from models.multitask import MLPBaseline,GRUBaseline
def metrics(ys,ps,ds,dp):
 per={}
 for i,name in enumerate(LABELS):
  tp=sum(y==i and q==i for y,q in zip(ys,ps));fp=sum(y!=i and q==i for y,q in zip(ys,ps));fn=sum(y==i and q!=i for y,q in zip(ys,ps));pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn) if tp+fn else 0;f=2*pr*re/(pr+re) if pr+re else 0;per[name]={'precision':pr,'recall':re,'f1':f,'support':ys.count(i)}
 err=np.asarray(dp)-np.asarray(ds);return {'accuracy':sum(y==q for y,q in zip(ys,ps))/len(ys),'macro_f1':sum(v['f1'] for v in per.values())/3,'weighted_f1':sum(v['f1']*v['support'] for v in per.values())/len(ys),'emergency_recall':per['Emergency']['recall'],'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt((err**2).mean())),'per_class':per,'confusion_matrix':[[sum(y==i and q==j for y,q in zip(ys,ps)) for j in range(3)] for i in range(3)],'test_samples':len(ys)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['mlp','gru'],required=True);p.add_argument('--feature-set',required=True);a=p.parse_args();parts,_=audit(a.feature_set);ck=torch.load(f'checkpoints/best_{a.model}_v21_{a.feature_set}.pt',map_location='cpu',weights_only=False);mean=np.asarray(ck['mean'],dtype=np.float32);std=np.asarray(ck['std'],dtype=np.float32);features=ck['features'];is_gru=a.model=='gru';dataset=Sequences if is_gru else Tabular;kwargs={'length':ck['sequence_length']} if is_gru else {};test=dataset(parts['test'],mean,std,features,**kwargs);loader=DataLoader(test,batch_size=256);model=(GRUBaseline(len(features)) if is_gru else MLPBaseline(len(features)));model.load_state_dict(ck['state_dict']);model.eval();ys=[];ps=[];ds=[];dp=[]
 with torch.no_grad():
  for x,y,d in loader:
   z,q=model(x);ys+=y.tolist();ps+=z.argmax(1).tolist();ds+=d.tolist();dp+=q.tolist()
 out=metrics(ys,ps,ds,dp);out['feature_set']=a.feature_set;root=Path('results/v21');root.mkdir(exist_ok=True);name=f'{a.model}_{a.feature_set}';(root/f'{name}_metrics.json').write_text(json.dumps(out,indent=2),encoding='utf-8');meta=endpoint_rows(parts['test'],ck['sequence_length']) if is_gru else parts['test']
 with open(root/f'{name}_test_predictions.csv','w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['scenario_id','time_s','true_risk','predicted_risk','true_distance','predicted_distance']);w.writerows([[r['scenario_id'],r['time_s'],LABELS[y],LABELS[q],d,z] for r,y,q,d,z in zip(meta,ys,ps,ds,dp)])
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
