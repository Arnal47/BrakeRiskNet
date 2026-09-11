"""Feature and timestep masking explanations for frozen GRU and Transformer checkpoints."""
import csv, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from training.data_v21 import audit
from training.v4_common import frozen_model, test_sequences, batch_predict, classification_metrics

ROOT=Path('results/v4')


def main():
    parts,_=audit('without_brake_state'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ROOT.mkdir(parents=True,exist_ok=True)
    feature_rows=[]; time_rows=[]; heat={}
    for name in ('gru','transformer'):
        model, checkpoint=frozen_model(name, device); data=test_sequences(parts, checkpoint)
        base_logits, base_distance, labels, truth=batch_predict(model,data,device); base=classification_metrics(labels,base_logits.argmax(1),truth,base_distance)
        losses=[]
        for feature_index,feature in enumerate(checkpoint['features']):
            logits,distance,_,_=batch_predict(model,data,device,lambda x,i=feature_index: x.clone().index_fill_(2,torch.tensor([i],device=x.device),0.0))
            score=classification_metrics(labels,logits.argmax(1),truth,distance); feature_rows.append({'model':name,'feature':feature,'masked_macro_f1':score['macro_f1'],'macro_f1_drop':base['macro_f1']-score['macro_f1'],'masked_distance_mae':score['distance_mae']})
        for step in range(checkpoint['sequence_length']):
            def mask_step(x,i=step): out=x.clone();out[:,i,:]=0.;return out
            logits,distance,_,_=batch_predict(model,data,device,mask_step); score=classification_metrics(labels,logits.argmax(1),truth,distance); drop=base['macro_f1']-score['macro_f1']; time_rows.append({'model':name,'timestep_from_history_start':step,'masked_macro_f1':score['macro_f1'],'macro_f1_drop':drop,'masked_distance_mae':score['distance_mae']});losses.append(drop)
        heat[name]=losses
    for path,rows in ((ROOT/'feature_masking.csv',feature_rows),(ROOT/'timestep_masking.csv',time_rows)):
        with open(path,'w',newline='',encoding='utf-8') as handle: writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.figure(figsize=(10,3));plt.imshow(np.asarray([heat['gru'],heat['transformer']]),cmap='magma',aspect='auto');plt.colorbar(label='Macro F1 drop after masking');plt.yticks([0,1],['GRU','Transformer']);plt.xticks(range(10),[f't-{9-i}' if i<9 else 't' for i in range(10)]);plt.xlabel('Masked history timestep');plt.title('Temporal masking sensitivity (frozen models)');plt.tight_layout();plt.savefig(ROOT/'temporal_masking_heatmap.png',dpi=160);plt.close()
    print(json.dumps({'feature_rows':feature_rows,'time_rows':time_rows},indent=2))


if __name__=='__main__': main()
