"""生成 V2.1 固定测试集模型对比与主模型混淆矩阵。"""
import csv,json
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
LABELS=['Safe','Warning','Emergency']
def draw(name):
 metric=json.loads(Path(f'results/v21/{name}_metrics.json').read_text());m=np.asarray(metric['confusion_matrix'])
 plt.figure(figsize=(5,4));plt.imshow(m,cmap='Blues');plt.xticks(range(3),LABELS);plt.yticks(range(3),LABELS);plt.xlabel('Predicted');plt.ylabel('True');plt.title(f'{name} confusion matrix')
 for i in range(3):
  for j in range(3):plt.text(j,i,str(m[i,j]),ha='center',va='center')
 plt.tight_layout();plt.savefig(f'results/v21/{name}_confusion_matrix.png',dpi=160);plt.close()
def row(label,m):return {'model':label,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'emergency_recall':m['emergency_recall'],'distance_mae':m['mae'],'distance_rmse':m['rmse'],'test_samples':m['test_samples']}
def main():
 root=Path('results/v21'); rows=[row('Physics Baseline',json.loads((root/'physics_metrics.json').read_text()))]
 for name,label in [('mlp_without_brake_state','MLP (primary: no brake_state)'),('gru_without_brake_state','GRU (primary: no brake_state)'),('mlp_without_ego_acceleration','MLP ablation: no ego_acceleration'),('mlp_without_both','MLP ablation: no brake_state & no ego_acceleration')]:
  metric=json.loads((root/f'{name}_metrics.json').read_text());rows.append(row(label,metric));draw(name)
 with open(root/'model_comparison.csv','w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
