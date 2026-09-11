"""Benchmark every V3 comparison model and record the exact runtime context."""
import json, platform, sys, time
from pathlib import Path
import numpy as np
import torch
from training.data_v21 import audit, fit_scaler, Tabular, Sequences, FEATURE_SETS
from models.multitask import MLPBaseline, GRUBaseline, TCNCausalBaseline, TransformerCausalBaseline
from baselines.physics_baseline_v15 import predict as physics_predict


ROOT = Path('results/v3')


def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def neural_latency(model, example, device, repeats=100, warmup=20):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup): model(example)
        synchronize(device); started = time.perf_counter()
        for _ in range(repeats): model(example)
        synchronize(device)
    return (time.perf_counter() - started) * 1000 / repeats


def physics_latency(row, repeats=1000, warmup=50):
    for _ in range(warmup): physics_predict(row)
    started = time.perf_counter()
    for _ in range(repeats): physics_predict(row)
    return (time.perf_counter() - started) * 1000 / repeats


def load_checkpoint(path, factory, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = factory(len(checkpoint['features'])).to(device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    return checkpoint, model


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    parts, _ = audit('without_brake_state'); features = FEATURE_SETS['without_brake_state']
    mean, std = fit_scaler(parts['train'], features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tabular = Tabular(parts['test'], mean, std, features)[0][0].unsqueeze(0).to(device)
    sequences = Sequences(parts['test'], mean, std, features, 10)[0][0].unsqueeze(0).to(device)
    models = {
        'mlp': (MLPBaseline, 'checkpoints/best_mlp_v21_without_brake_state.pt', tabular),
        'gru': (GRUBaseline, 'checkpoints/best_gru_v21_without_brake_state.pt', sequences),
        'tcn': (TCNCausalBaseline, 'checkpoints/best_tcn_v3_without_brake_state_len10.pt', sequences),
        'transformer': (TransformerCausalBaseline, 'checkpoints/best_transformer_v3_without_brake_state_len10.pt', sequences),
    }
    results = {
        'physics': {'parameters': 0, 'latency_ms_batch_1': physics_latency(parts['test'][9]), 'latency_ms_batch_256': None, 'latency_device': 'cpu (Python scalar baseline)'},
    }
    for name, (factory, path, example) in models.items():
        _, model = load_checkpoint(path, factory, device)
        results[name] = {'parameters': sum(parameter.numel() for parameter in model.parameters()), 'latency_ms_batch_1': neural_latency(model, example, device), 'latency_ms_batch_256': neural_latency(model, example.repeat(256, *([1] * (example.ndim - 1))), device), 'latency_device': str(device)}
    metadata = {
        'python_version': sys.version.split()[0], 'platform': platform.platform(), 'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'benchmark_protocol': '20 warm-up iterations; mean wall-clock latency over 100 neural forwards, synchronized on CUDA. Physics is a scalar Python implementation over 1,000 calls.',
        'models': results,
    }
    (ROOT / 'deployment_metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__': main()
