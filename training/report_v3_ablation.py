"""Summarize validation-only V3 sequence-length ablations; never reads test outputs."""
import argparse, csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--model', choices=['tcn', 'transformer'], required=True)
    parser.add_argument('--lengths', nargs='+', type=int, default=[5, 10, 20]); args = parser.parse_args()
    rows = []
    for length in args.lengths:
        path = Path(f'results/v3/{args.model}_without_brake_state_len{length}_history.csv')
        with open(path, newline='', encoding='utf-8') as handle: history = list(csv.DictReader(handle))
        best = min(history, key=lambda row: float(row['validation_loss']))
        rows.append({'model': args.model, 'sequence_length': length, 'best_epoch': best['epoch'], 'validation_loss': best['validation_loss'], 'validation_accuracy': best['validation_accuracy']})
    target = Path(f'results/v3/{args.model}_validation_length_ablation.csv')
    with open(target, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(rows)


if __name__ == '__main__': main()
