"""Evaluate V3 checkpoints once on the fixed V2.1 test scenarios."""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.data_v21 import audit, Sequences, LABELS, endpoint_rows
from training.evaluate_v21 import metrics
from models.multitask import TCNCausalBaseline, TransformerCausalBaseline


def build(name, input_dim):
    return TCNCausalBaseline(input_dim) if name == 'tcn' else TransformerCausalBaseline(input_dim)


def latency(model, example, device):
    # Warm-up avoids recording one-time kernel/library initialization.
    with torch.no_grad():
        for _ in range(20): model(example)
        if device.type == 'cuda': torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100): model(example)
        if device.type == 'cuda': torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / 100


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--model', choices=['tcn', 'transformer'], required=True)
    parser.add_argument('--feature-set', default='without_brake_state'); parser.add_argument('--sequence-length', type=int, default=10)
    args = parser.parse_args(); parts, _ = audit(args.feature_set)
    checkpoint_path = Path(f'checkpoints/best_{args.model}_v3_{args.feature_set}_len{args.sequence_length}.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = build(args.model, len(checkpoint['features'])).to(device); model.load_state_dict(checkpoint['state_dict']); model.eval()
    test = Sequences(parts['test'], np.asarray(checkpoint['mean'], dtype=np.float32), np.asarray(checkpoint['std'], dtype=np.float32), checkpoint['features'], checkpoint['sequence_length'])
    loader = DataLoader(test, batch_size=256); ys=[]; ps=[]; ds=[]; dp=[]
    with torch.no_grad():
        for x, y, d in loader:
            logits, prediction = model(x.to(device)); ys += y.tolist(); ps += logits.argmax(1).cpu().tolist(); ds += d.tolist(); dp += prediction.cpu().tolist()
    result = metrics(ys, ps, ds, dp)
    example = test[0][0].unsqueeze(0).to(device)
    result.update({'model': args.model, 'feature_set': args.feature_set, 'sequence_length': checkpoint['sequence_length'], 'parameters': sum(p.numel() for p in model.parameters()), 'latency_ms_batch_1': latency(model, example, device), 'latency_ms_batch_256': latency(model, example.repeat(256, 1, 1), device), 'latency_device': str(device)})
    root = Path('results/v3'); root.mkdir(parents=True, exist_ok=True); name = f'{args.model}_{args.feature_set}_len{args.sequence_length}'
    (root / f'{name}_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    metadata = endpoint_rows(parts['test'], checkpoint['sequence_length'])
    with open(root / f'{name}_test_predictions.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(['scenario_id','time_s','true_risk','predicted_risk','true_distance','predicted_distance'])
        writer.writerows([[r['scenario_id'], r['time_s'], LABELS[y], LABELS[p], d, predicted] for r, y, p, d, predicted in zip(metadata, ys, ps, ds, dp)])
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()





