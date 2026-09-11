"""只对固定 V1.5 test split 评估 checkpoint，并保存完整分类/回归结果。"""
import argparse,csv,json,math
from collections import Counter
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from training.data import audit,fit_scaler,Tabular,Sequences,LABELS
from models.multitask import MLPBaseline,GRUBaseline
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['mlp','gru'],required=True);a=p.parse_args();parts,_=audit();ck=torch.load(f'checkpoints/best_{a.model}_v2.pt',map_location='cpu',weights_only=False);dataset=Sequences if a.model=='gru' else Tabular;kwargs={'length':ck['sequence_length']} if a.model=='gru' else {};test=dataset(parts['test'],ck['mean'],ck['std'],**kwargs);loader=DataLoader(test,batch_size=256);model=(GRUBaseline(len(ck['mean'])) if a.model=='gru' else MLPBaseline(len(ck['mean'])));model.load_state_dict(ck['state_dict']);model.eval();ys=[];ps=[];ds=[];dp=[]
 with torch.no_grad():
  for x,y,d in loader:
   z,q=model(x);ys+=y.tolist();ps+=z.argmax(1).tolist();ds+=d.tolist();dp+=q.tolist()
 per={};
 for i,name in enumerate(LABELS):
  tp=sum(y==i and q==i for y,q in zip(ys,ps));fp=sum(y!=i and q==i for y,q in zip(ys,ps));fn=sum(y==i and q!=i for y,q in zip(ys,ps));pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn) if tp+fn else 0;f=2*pr*re/(pr+re) if pr+re else 0;per[name]={'precision':pr,'recall':re,'f1':f,'support':ys.count(i)}
 acc=sum(y==q for y,q in zip(ys,ps))/len(ys);macro=sum(v['f1'] for v in per.values())/3;weighted=sum(v['f1']*v['support'] for v in per.values())/len(ys);err=np.array(dp)-np.array(ds);metrics={'accuracy':acc,'macro_f1':macro,'weighted_f1':weighted,'emergency_recall':per['Emergency']['recall'],'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt((err**2).mean())),'per_class':per,'test_samples':len(ys)};Path('results/v2').mkdir(exist_ok=True);Path(f'results/v2/{a.model}_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');
 with open(f'results/v2/{a.model}_test_predictions.csv','w',newline='',encoding='utf-8') as f:w=csv.writer(f);w.writerow(['true_risk','predicted_risk','true_distance','predicted_distance']);w.writerows([[LABELS[y],LABELS[q],d,r] for y,q,d,r in zip(ys,ps,ds,dp)])
 print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()

