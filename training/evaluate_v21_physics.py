"""V2.1 Physics Baseline：只读固定 test split，不重新划分数据。"""
import csv,json,math
from collections import Counter
from pathlib import Path
from baselines.physics_baseline_v15 import predict
LABELS=['Safe','Warning','Emergency']
def score(rows):
 pairs=[(r,*predict(r)) for r in rows]; per={}; f1=[]
 for label in LABELS:
  tp=sum(r['risk_level']==label and p==label for r,p,_ in pairs);fp=sum(r['risk_level']!=label and p==label for r,p,_ in pairs);fn=sum(r['risk_level']==label and p!=label for r,p,_ in pairs);pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn) if tp+fn else 0;v=2*pr*re/(pr+re) if pr+re else 0;per[label]={'precision':pr,'recall':re,'f1':v,'support':sum(r['risk_level']==label for r,_,_ in pairs)};f1.append(v)
 err=[d-float(r['stopping_distance']) for r,_,d in pairs]; acc=sum(r['risk_level']==p for r,p,_ in pairs)/len(pairs)
 return {'accuracy':acc,'macro_f1':sum(f1)/3,'weighted_f1':sum(per[k]['f1']*per[k]['support'] for k in LABELS)/len(pairs),'emergency_recall':per['Emergency']['recall'],'mae':sum(abs(x) for x in err)/len(err),'rmse':math.sqrt(sum(x*x for x in err)/len(err)),'per_class':per,'test_samples':len(pairs)},pairs
def main():
 with open('data/v21_splits/test.csv',newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 metrics,pairs=score(rows);metrics['risk_distribution']=dict(Counter(r['risk_level'] for r in rows));Path('results/v21').mkdir(parents=True,exist_ok=True);Path('results/v21/physics_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
 with open('results/v21/physics_test_predictions.csv','w',newline='',encoding='utf-8') as f:w=csv.writer(f);w.writerow(['true_risk','predicted_risk','true_distance','predicted_distance']);w.writerows([[r['risk_level'],p,r['stopping_distance'],d] for r,p,d in pairs])
 print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
