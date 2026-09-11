"""V1.5 基线评估：总体、类别分布与困难条件切片。"""
import csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
from baselines.physics_baseline_v15 import predict
from utils.split import split_by_scenario,assert_no_scenario_leakage
LABELS=['Safe','Warning','Emergency']
def score(rows):
    pairs=[(r,*predict(r)) for r in rows]; accuracy=sum(r['risk_level']==p for r,p,_ in pairs)/len(pairs); f1=[]; per={}
    for label in LABELS:
        tp=sum(r['risk_level']==label and p==label for r,p,_ in pairs); fp=sum(r['risk_level']!=label and p==label for r,p,_ in pairs); fn=sum(r['risk_level']==label and p!=label for r,p,_ in pairs); pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0; value=2*pr*re/(pr+re) if pr+re else 0; f1.append(value); per[label]={'precision':pr,'recall':re,'f1':value,'support':sum(r['risk_level']==label for r,_,_ in pairs)}
    err=[d-float(r['stopping_distance']) for r,_,d in pairs]
    return {'accuracy':accuracy,'macro_f1':sum(f1)/3,'mae':sum(abs(x) for x in err)/len(err),'rmse':math.sqrt(sum(x*x for x in err)/len(err)),'per_class':per},pairs
def main():
    with open('data/synthetic_driving_v15.csv',newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    splits,counts=split_by_scenario(rows,seed=42); overlap=assert_no_scenario_leakage(splits); metrics,pairs=score(splits['test']); distribution=Counter(r['risk_level'] for r in rows)
    slices={}
    for name,part in {'sudden_braking':[r for r in splits['test'] if r['scenario_type']=='sudden_lead_braking'],'wet_low_adhesion':[r for r in splits['test'] if r['road_condition']=='wet_low_adhesion'],'high_speed':[r for r in splits['test'] if r['scenario_type']=='high_speed_approach'],'sensor_delay':[r for r in splits['test'] if int(r['sensor_delay_ms'])>0]}.items(): slices[name]=score(part)[0] if part else None
    summary={'v1_reference':{'accuracy':.9929348,'macro_f1':.9924168,'mae':.2108989,'rmse':.2983203},'v15':metrics,'risk_distribution':dict(distribution),'split_scenarios':counts,'scenario_overlap':overlap,'condition_slices':slices}
    Path('results').mkdir(exist_ok=True); Path('results/v15_baseline_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with open('results/v15_predictions.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['scenario_id','time_s','scenario_type','road_condition','true_risk','predicted_risk','true_stopping_distance','predicted_stopping_distance']); [w.writerow([r['scenario_id'],r['time_s'],r['scenario_type'],r['road_condition'],r['risk_level'],p,r['stopping_distance'],f'{d:.3f}']) for r,p,d in pairs]
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
