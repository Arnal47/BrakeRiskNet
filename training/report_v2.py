"""汇总 Physics / MLP / GRU 的固定测试结果并绘制混淆矩阵。"""
import csv,json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
LABELS=['Safe','Warning','Emergency']
def confusion(model):
    m=np.zeros((3,3),dtype=int)
    with open(f'results/v2/{model}_test_predictions.csv',newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):m[LABELS.index(r['true_risk']),LABELS.index(r['predicted_risk'])]+=1
    plt.figure(figsize=(5,4));plt.imshow(m,cmap='Blues');plt.xticks(range(3),LABELS);plt.yticks(range(3),LABELS);plt.xlabel('Predicted');plt.ylabel('True');plt.title(f'{model.upper()} confusion matrix');
    for i in range(3):
        for j in range(3):plt.text(j,i,str(m[i,j]),ha='center',va='center')
    plt.tight_layout();plt.savefig(f'results/v2/{model}_confusion_matrix.png',dpi=160);plt.close()
def main():
    root=Path('results');v15=json.loads((root/'v15_baseline_metrics.json').read_text());physics=v15['v15']; rows=[]
    rows.append({'model':'Physics Baseline','accuracy':physics['accuracy'],'macro_f1':physics['macro_f1'],'emergency_recall':physics['per_class']['Emergency']['recall'],'distance_mae':physics['mae'],'distance_rmse':physics['rmse'],'test_scenarios':45,'test_samples':sum(1 for _ in open(root/'v15_predictions.csv', encoding='utf-8'))-1})
    for model in ['mlp','gru']:
        metric=json.loads((root/f'v2/{model}_metrics.json').read_text()); rows.append({'model':model.upper(),'accuracy':metric['accuracy'],'macro_f1':metric['macro_f1'],'emergency_recall':metric['emergency_recall'],'distance_mae':metric['mae'],'distance_rmse':metric['rmse'],'test_scenarios':45,'test_samples':metric['test_samples']});confusion(model)
    with open(root/'v2/model_comparison.csv','w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(rows)
if __name__=='__main__':main()

