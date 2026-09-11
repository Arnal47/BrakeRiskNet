"""训练 MLP 或 GRU；test 集从不在此脚本中读取。"""
import argparse,csv,json,random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from training.data import audit,fit_scaler,Tabular,Sequences,LABELS,LABEL_MAP
from models.multitask import MLPBaseline,GRUBaseline

def evaluate(model,loader,device):
    model.eval(); loss=0; correct=total=0
    with torch.no_grad():
        for x,y,d in loader:
            logits,pred=model(x.to(device)); y,d=y.to(device),d.to(device); loss+=torch.nn.functional.cross_entropy(logits,y).item()*len(y)+.05*torch.nn.functional.smooth_l1_loss(pred,d).item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); total+=len(y)
    return loss/total,correct/total
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['mlp','gru'],required=True);p.add_argument('--epochs',type=int,default=35);p.add_argument('--sequence-length',type=int,default=10);a=p.parse_args(); seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    parts,audit_summary=audit(); mean,std=fit_scaler(parts['train']); Path('results/v2').mkdir(parents=True,exist_ok=True); np.savez('results/v2/train_scaler.npz',mean=mean,std=std)
    dataset=Sequences if a.model=='gru' else Tabular; kwargs={'length':a.sequence_length} if a.model=='gru' else {}; train=dataset(parts['train'],mean,std,**kwargs);val=dataset(parts['validation'],mean,std,**kwargs)
    train_loader=DataLoader(train,batch_size=128,shuffle=True);val_loader=DataLoader(val,batch_size=256);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=(GRUBaseline(len(mean)) if a.model=='gru' else MLPBaseline(len(mean))).to(device)
    counts=np.bincount([LABEL_MAP[r['risk_level']] for r in parts['train']],minlength=3); weights=torch.tensor(counts.sum()/(3*counts),dtype=torch.float32,device=device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); ce=torch.nn.CrossEntropyLoss(weight=weights); best=1e9;wait=0; history=[]; Path('checkpoints').mkdir(exist_ok=True)
    for epoch in range(1,a.epochs+1):
        model.train();running=0
        for x,y,d in train_loader:
            x,y,d=x.to(device),y.to(device),d.to(device);opt.zero_grad();logits,pred=model(x);loss=ce(logits,y)+.05*torch.nn.functional.smooth_l1_loss(pred,d);loss.backward();opt.step();running+=loss.item()*len(y)
        val_loss,val_acc=evaluate(model,val_loader,device);history.append({'epoch':epoch,'train_loss':running/len(train),'validation_loss':val_loss,'validation_accuracy':val_acc});print(a.model,history[-1])
        if val_loss<best: best=val_loss;wait=0;torch.save({'state_dict':model.state_dict(),'mean':mean,'std':std,'sequence_length':a.sequence_length,'features':__import__('training.data',fromlist=['FEATURES']).FEATURES},f'checkpoints/best_{a.model}_v2.pt')
        else: wait+=1
        if wait>=7:break
    with open(f'results/v2/{a.model}_history.csv','w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=history[0].keys());w.writeheader();w.writerows(history)
    print({'device':str(device),'train_samples':len(train),'validation_samples':len(val),'class_weights':weights.tolist(),'audit':audit_summary})
if __name__=='__main__':main()

