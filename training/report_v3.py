"""Produce the single aligned-endpoint V3 comparison and visual artefacts."""
import csv, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from training.report_v21 import calculate


ROOT = Path('results/v3')
V21 = Path('results/v21')
SPECS = [
    ('physics', 'Physics Baseline', V21 / 'physics_test_predictions.csv', None),
    ('mlp', 'MLP', V21 / 'mlp_without_brake_state_test_predictions.csv', None),
    ('gru', 'GRU', V21 / 'gru_without_brake_state_test_predictions.csv', V21 / 'gru_without_brake_state_metrics.json'),
    ('tcn', 'Causal TCN', ROOT / 'tcn_without_brake_state_len10_test_predictions.csv', ROOT / 'tcn_without_brake_state_len10_metrics.json'),
    ('transformer', 'Lightweight Transformer', ROOT / 'transformer_without_brake_state_len10_test_predictions.csv', ROOT / 'transformer_without_brake_state_len10_metrics.json'),
]


def load(path):
    with open(path, newline='', encoding='utf-8') as handle: return list(csv.DictReader(handle))


def count_parameters_from_checkpoint(model):
    # Existing V2.1 checkpoints are intentionally untracked, so their count is structural.
    return {'mlp': 2980, 'gru': 16804}.get(model, '')


def draw_confusion(name, rows):
    labels = ['Safe', 'Warning', 'Emergency']; matrix = np.zeros((3, 3), dtype=int)
    for row in rows: matrix[labels.index(row['true_risk']), labels.index(row['predicted_risk'])] += 1
    plt.figure(figsize=(5, 4)); plt.imshow(matrix, cmap='Blues'); plt.xticks(range(3), labels); plt.yticks(range(3), labels)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'{name} aligned-endpoint confusion matrix')
    for i in range(3):
        for j in range(3): plt.text(j, i, str(matrix[i, j]), ha='center', va='center')
    plt.tight_layout(); plt.savefig(ROOT / f'{name}_confusion_matrix.png', dpi=160); plt.close()


def draw_loss_curves():
    plt.figure(figsize=(7, 4))
    for model in ('tcn', 'transformer'):
        path = ROOT / f'{model}_without_brake_state_len10_history.csv'
        if not path.exists(): continue
        rows = load(path)
        epochs = [int(row['epoch']) for row in rows]
        plt.plot(epochs, [float(row['train_loss']) for row in rows], label=f'{model} train')
        plt.plot(epochs, [float(row['validation_loss']) for row in rows], '--', label=f'{model} validation')
    plt.xlabel('Epoch'); plt.ylabel('Multitask loss'); plt.title('V3 training history'); plt.legend(); plt.tight_layout()
    plt.savefig(ROOT / 'v3_loss_curves.png', dpi=160); plt.close()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    endpoint = {(r['scenario_id'], r['time_s']) for r in load(V21 / 'gru_without_brake_state_test_predictions.csv')}
    output = []
    for key, label, prediction_path, metric_path in SPECS:
        rows = [r for r in load(prediction_path) if (r['scenario_id'], r['time_s']) in endpoint]
        metric = calculate(rows)
        extra = json.loads(metric_path.read_text(encoding='utf-8')) if metric_path else {}
        output.append({'model': label, 'accuracy': metric['accuracy'], 'macro_f1': metric['macro_f1'], 'emergency_recall': metric['emergency_recall'], 'distance_mae': metric['distance_mae'], 'distance_rmse': metric['distance_rmse'], 'parameters': extra.get('parameters', count_parameters_from_checkpoint(key)), 'latency_ms_batch_1_cpu': extra.get('latency_ms_batch_1_cpu', ''), 'latency_ms_batch_256_cpu': extra.get('latency_ms_batch_256_cpu', ''), 'test_samples': metric['test_samples']})
        draw_confusion(key, rows)
    fields = list(output[0])
    with open(ROOT / 'aligned_endpoint_comparison.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output)
    (ROOT / 'aligned_endpoint_comparison.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    draw_loss_curves()
    print(json.dumps(output, indent=2))


if __name__ == '__main__': main()
