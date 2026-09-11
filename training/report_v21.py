"""生成 all-frame 附表和以 GRU endpoint 为准的公平主表。"""
import csv,json,math
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
LABELS=['Safe','Warning','Emergency']
def calculate(rows):
 ys=[LABELS.index(r['true_risk']) for r in rows];ps=[LABELS.index(r['predicted_risk']) for r in rows];ds=np.asarray([float(r['true_distance']) for r in rows]);dp=np.asarray([float(r['predicted_distance']) for r in rows]);per=[]
 for i in range(3):
  tp=sum(y==i and p==i for y,p in zip(ys,ps));fp=sum(y!=i and p==i for y,p in zip(ys,ps));fn=sum(y==i and p!=i for y,p in zip(ys,ps));pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn) if tp+fn else 0;per.append((2*pr*re/(pr+re) if pr+re else 0,re))
 err=dp-ds;return {'accuracy':sum(y==p for y,p in zip(ys,ps))/len(ys),'macro_f1':sum(x[0] for x in per)/3,'emergency_recall':per[2][1],'distance_mae':float(np.abs(err).mean()),'distance_rmse':float(np.sqrt((err**2).mean())),'test_samples':len(ys)}
def load(name):
 with open(f'results/v21/{name}_test_predictions.csv',newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(path,rows):
 with open(path,'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['model','accuracy','macro_f1','emergency_recall','distance_mae','distance_rmse','test_samples']);w.writeheader();w.writerows(rows)
def draw(name):
 m=np.asarray(json.loads(Path(f'results/v21/{name}_metrics.json').read_text())['confusion_matrix']);plt.figure(figsize=(5,4));plt.imshow(m,cmap='Blues');plt.xticks(range(3),LABELS);plt.yticks(range(3),LABELS);plt.xlabel('Predicted');plt.ylabel('True');plt.title(f'{name} confusion matrix')
 for i in range(3):
  for j in range(3):plt.text(j,i,str(m[i,j]),ha='center',va='center')
 plt.tight_layout();plt.savefig(f'results/v21/{name}_confusion_matrix.png',dpi=160);plt.close()
def main():
 root=Path('results/v21'); names=[('physics','Physics Baseline'),('mlp_without_brake_state','MLP (primary: no brake_state)'),('gru_without_brake_state','GRU (primary: no brake_state)'),('mlp_without_ego_acceleration','MLP ablation: no ego_acceleration'),('mlp_without_both','MLP ablation: no brake_state & no ego_acceleration')]; all_rows=[dict(model=label,**calculate(load(name))) for name,label in names];write(root/'all_frame_comparison.csv',all_rows)
 # GRU 预测文件已经只含每场景第 10 步及之后的端点；以其 (scenario_id,time_s) 作为公平交集。
 endpoint={(r['scenario_id'],r['time_s']) for r in load('gru_without_brake_state')};aligned=[]
 for name,label in names[:3]:aligned.append(dict(model=label,**calculate([r for r in load(name) if (r['scenario_id'],r['time_s']) in endpoint])))
 write(root/'aligned_endpoint_comparison.csv',aligned)
 for name,_ in names[1:]:draw(name)
 print(json.dumps({'aligned_endpoint':aligned,'all_frame':all_rows},indent=2))
if __name__=='__main__':main()
