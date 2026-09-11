"""训练 V2.1 MLP/GRU；仅使用固定 train/validation，test 只由评估脚本读取。"""
import argparse, csv, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from training.data_v21 import audit, fit_scaler, Tabular, Sequences, FEATURE_SETS, LABEL_MAP
from models.multitask import MLPBaseline, GRUBaseline

def evaluate(model, loader, device, ce):
    model.eval(); total = correct = 0; loss = 0.
    with torch.no_grad():
        for x,y,d in loader:
            logits,pred=model(x.to(device)); y,d=y.to(device),d.to(device)
            loss += (ce(logits,y)+.05*torch.nn.functional.smooth_l1_loss(pred,d)).item()*len(y)
            correct += (logits.argmax(1)==y).sum().item(); total += len(y)
    return loss/total, correct/total

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['mlp','gru'],required=True); p.add_argument('--feature-set',choices=FEATURE_SETS,default='without_brake_state')
    p.add_argument('--epochs',type=int,default=35); p.add_argument('--sequence-length',type=int,default=10); a=p.parse_args()
    seed=42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    parts, summary = audit(a.feature_set); features=FEATURE_SETS[a.feature_set]; mean,std=fit_scaler(parts['train'],features)
    Path('results/v21').mkdir(parents=True,exist_ok=True); np.savez(f'results/v21/{a.model}_{a.feature_set}_scaler.npz',mean=mean,std=std)
    dataset=Sequences if a.model=='gru' else Tabular; kwargs={'length':a.sequence_length} if a.model=='gru' else {}
    train=dataset(parts['train'],mean,std,features,**kwargs); val=dataset(parts['validation'],mean,std,features,**kwargs)
    train_loader=DataLoader(train,batch_size=128,shuffle=True); val_loader=DataLoader(val,batch_size=256)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=(GRUBaseline(len(features)) if a.model=='gru' else MLPBaseline(len(features))).to(device)
    # GRU 的权重必须来自实际 sequence endpoint 标签；MLP 则等于 train 行标签。
    counts=np.bincount([train[i][1].item() for i in range(len(train))],minlength=3); weights=torch.tensor(counts.sum()/(3*counts),dtype=torch.float32,device=device)
    ce=torch.nn.CrossEntropyLoss(weight=weights); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=float('inf'); wait=0; history=[]; Path('checkpoints').mkdir(exist_ok=True)
    for epoch in range(1,a.epochs+1):
        model.train(); running=0.
        for x,y,d in train_loader:
            x,y,d=x.to(device),y.to(device),d.to(device); opt.zero_grad(); logits,pred=model(x)
            loss=ce(logits,y)+.05*torch.nn.functional.smooth_l1_loss(pred,d); loss.backward(); opt.step(); running+=loss.item()*len(y)
        val_loss,val_acc=evaluate(model,val_loader,device,ce); row={'epoch':epoch,'train_loss':running/len(train),'validation_loss':val_loss,'validation_accuracy':val_acc}; history.append(row); print(a.model,a.feature_set,row)
        if val_loss<best:
            best=val_loss;wait=0
            torch.save({'state_dict':model.state_dict(),'mean':mean.tolist(),'std':std.tolist(),'sequence_length':a.sequence_length,'features':features,'feature_set':a.feature_set},f'checkpoints/best_{a.model}_v21_{a.feature_set}.pt')
        else: wait+=1
        if wait>=7: break
    with open(f'results/v21/{a.model}_{a.feature_set}_history.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys());w.writeheader();w.writerows(history)
    print({'device':str(device),'train_samples':len(train),'validation_samples':len(val),'class_weights':weights.tolist(),'audit':summary})
if __name__=='__main__': main()

