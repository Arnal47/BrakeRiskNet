"""评估 Physics Baseline，并保存少量可复现结果。"""
import csv, json, math
from collections import Counter
from pathlib import Path
from baselines.physics_baseline import predict

LABELS = ['Safe','Warning','Emergency']
def main():
    with open('data/splits/test.csv', newline='', encoding='utf-8') as file: rows = list(csv.DictReader(file))
    pairs = [(row, *predict(row)) for row in rows]
    accuracy = sum(row['risk_level']==risk for row,risk,_ in pairs)/len(pairs)
    report = {}; macro_f1 = []
    for label in LABELS:
        tp=sum(r['risk_level']==label and p==label for r,p,_ in pairs); fp=sum(r['risk_level']!=label and p==label for r,p,_ in pairs); fn=sum(r['risk_level']==label and p!=label for r,p,_ in pairs)
        precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0
        report[label]={'precision':precision,'recall':recall,'f1':f1,'support':sum(r['risk_level']==label for r,_,_ in pairs)}; macro_f1.append(f1)
    errors=[pred-float(row['stopping_distance']) for row,_,pred in pairs]
    metrics={'risk_level_accuracy':accuracy,'macro_f1':sum(macro_f1)/len(macro_f1),'stopping_distance_mae':sum(abs(x) for x in errors)/len(errors),'stopping_distance_rmse':math.sqrt(sum(x*x for x in errors)/len(errors)),'per_class':report,'test_steps':len(rows),'test_scenarios':len({r['scenario_id'] for r in rows})}
    Path('results').mkdir(exist_ok=True); Path('results/physics_baseline_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    with open('results/physics_baseline_predictions.csv','w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file); writer.writerow(['scenario_id','time_s','true_risk','predicted_risk','true_stopping_distance','predicted_stopping_distance'])
        for row,risk,distance in pairs: writer.writerow([row['scenario_id'],row['time_s'],row['risk_level'],risk,row['stopping_distance'],f'{distance:.4f}'])
    print(json.dumps(metrics,indent=2))
if __name__ == '__main__': main()
