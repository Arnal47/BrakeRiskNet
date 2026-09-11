"""Frozen-model stress tests. Results are descriptive and never feed model selection."""
import csv, json
from pathlib import Path
import numpy as np
import torch
from training.data_v21 import audit
from training.v4_common import frozen_model, test_sequences, batch_predict, classification_metrics

ROOT=Path('results/v4')


def perturbation(name, features, std):
    index={feature:i for i,feature in enumerate(features)}
    if name == 'clean': return lambda x: x
    if name == 'sensor_noise':
        affected=[index[k] for k in ('ego_speed','lead_speed','distance','relative_speed','ttc')]
        scale=torch.tensor([1.2, 1.2, 2.0, 0.7, 0.25], dtype=torch.float32) / torch.tensor(std[affected], dtype=torch.float32)
        generator=torch.Generator().manual_seed(2026)
        return lambda x: x.index_add(2, torch.tensor(affected, device=x.device), torch.randn(x.shape[0], x.shape[1], len(affected), generator=generator, device=x.device)*scale.to(x.device))
    if name == 'additional_delay':
        def delayed(x):
            out=x.clone(); out[:, 1:, :]=x[:, :-1, :]; return out
        return delayed
    if name == 'dropout':
        affected=[index[k] for k in ('distance','relative_speed','ttc')]
        def dropped(x):
            out=x.clone(); out[:, -1, affected]=0.0; return out
        return dropped
    if name == 'low_friction':
        friction=index['road_friction']; delta=.25/std[friction]
        def low_mu(x):
            out=x.clone(); out[:,:,friction]-=delta; return out
        return low_mu
    raise ValueError(name)


def main():
    parts, audit_summary=audit('without_brake_state'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ROOT.mkdir(parents=True, exist_ok=True); rows=[]
    for model_name in ('gru','tcn','transformer'):
        model, checkpoint=frozen_model(model_name, device); data=test_sequences(parts, checkpoint)
        for condition in ('clean','sensor_noise','additional_delay','dropout','low_friction'):
            logits, distances, labels, truth=batch_predict(model, data, device, perturbation(condition, checkpoint['features'], np.asarray(checkpoint['std'])))
            row={'model':model_name, 'condition':condition, **classification_metrics(labels, logits.argmax(1), truth, distances), 'test_endpoints':len(labels), 'model_frozen':True}
            rows.append(row)
    fields=list(rows[0]);
    with open(ROOT/'robustness_stress_test.csv','w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (ROOT/'robustness_stress_test.json').write_text(json.dumps({'protocol':'Frozen V2.1/V3 checkpoints; fixed held-out endpoints; no retraining, selection, or hyperparameter changes. Noise is deterministic seed 2026. Dropout uses mean imputation (normalized zero).', 'audit':audit_summary, 'results':rows},indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
