"""Train V3 temporal baselines without changing the V2.1 data protocol."""
import argparse, csv, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.data_v21 import audit, fit_scaler, Sequences, FEATURE_SETS
from models.multitask import TCNCausalBaseline, TransformerCausalBaseline


def build(name, input_dim):
    return TCNCausalBaseline(input_dim) if name == 'tcn' else TransformerCausalBaseline(input_dim)


def evaluate(model, loader, device, ce):
    model.eval(); total = correct = 0; loss = 0.
    with torch.no_grad():
        for x, y, d in loader:
            logits, prediction = model(x.to(device)); y, d = y.to(device), d.to(device)
            loss += (ce(logits, y) + .05 * torch.nn.functional.smooth_l1_loss(prediction, d)).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item(); total += len(y)
    return loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['tcn', 'transformer'], required=True)
    parser.add_argument('--feature-set', choices=FEATURE_SETS, default='without_brake_state')
    parser.add_argument('--sequence-length', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=35)
    args = parser.parse_args()
    if args.sequence_length < 2: raise ValueError('sequence length must be at least two')
    seed = 42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    parts, summary = audit(args.feature_set); features = FEATURE_SETS[args.feature_set]
    mean, std = fit_scaler(parts['train'], features)
    train = Sequences(parts['train'], mean, std, features, args.sequence_length)
    validation = Sequences(parts['validation'], mean, std, features, args.sequence_length)
    train_loader = DataLoader(train, batch_size=128, shuffle=True); validation_loader = DataLoader(validation, batch_size=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = build(args.model, len(features)).to(device)
    counts = np.bincount([train[i][1].item() for i in range(len(train))], minlength=3)
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    ce = torch.nn.CrossEntropyLoss(weight=weights); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    root = Path('results/v3'); root.mkdir(parents=True, exist_ok=True); Path('checkpoints').mkdir(exist_ok=True)
    best = float('inf'); wait = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.
        for x, y, d in train_loader:
            x, y, d = x.to(device), y.to(device), d.to(device); optimizer.zero_grad()
            logits, prediction = model(x); loss = ce(logits, y) + .05 * torch.nn.functional.smooth_l1_loss(prediction, d)
            loss.backward(); optimizer.step(); running += loss.item() * len(y)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device, ce)
        row = {'epoch': epoch, 'train_loss': running / len(train), 'validation_loss': validation_loss, 'validation_accuracy': validation_accuracy}
        history.append(row); print(args.model, row)
        if validation_loss < best:
            best = validation_loss; wait = 0
            torch.save({'state_dict': model.state_dict(), 'mean': mean.tolist(), 'std': std.tolist(), 'sequence_length': args.sequence_length, 'features': features, 'feature_set': args.feature_set, 'model': args.model, 'validation_loss': best}, f'checkpoints/best_{args.model}_v3_{args.feature_set}_len{args.sequence_length}.pt')
        else: wait += 1
        if wait >= 7: break
    with open(root / f'{args.model}_{args.feature_set}_len{args.sequence_length}_history.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    print({'device': str(device), 'train_endpoints': len(train), 'validation_endpoints': len(validation), 'class_weights': weights.tolist(), 'audit': summary})


if __name__ == '__main__': main()
