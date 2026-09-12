"""Shared frozen-model utilities for V4 analysis; these never train or tune a model."""
from pathlib import Path
import numpy as np
import torch
from training.data_v21 import FEATURE_SETS, Sequences, fit_scaler
from models.multitask import GRUBaseline, TCNCausalBaseline, TransformerCausalBaseline

MODEL_SPECS = {
    'gru': (GRUBaseline, 'checkpoints/best_gru_v21_without_brake_state.pt'),
    'tcn': (TCNCausalBaseline, 'checkpoints/best_tcn_v3_without_brake_state_len10.pt'),
    'transformer': (TransformerCausalBaseline, 'checkpoints/best_transformer_v3_without_brake_state_len10.pt'),
}


def frozen_model(name, device):
    factory, path = MODEL_SPECS[name]
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = factory(len(checkpoint['features'])).to(device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    return model, checkpoint


def test_sequences(parts, checkpoint):
    features = checkpoint['features']
    return Sequences(parts['test'], np.asarray(checkpoint['mean'], dtype=np.float32), np.asarray(checkpoint['std'], dtype=np.float32), features, checkpoint['sequence_length'])


def batch_predict(model, dataset, device, transform=None, batch_size=256):
    logits_all=[]; distances=[]; labels=[]; truth=[]
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            examples=[dataset[i] for i in range(start, min(start+batch_size, len(dataset)))]
            x=torch.stack([item[0] for item in examples]).to(device)
            if transform: x=transform(x)
            logits, distance=model(x); logits_all.append(logits.cpu()); distances.append(distance.cpu())
            labels.extend(item[1].item() for item in examples); truth.extend(item[2].item() for item in examples)
    return torch.cat(logits_all).numpy(), np.concatenate([value.numpy() for value in distances]), np.asarray(labels), np.asarray(truth)


def classification_metrics(labels, predictions, distance_truth, distance_predictions):
    f1=[]
    for cls in range(3):
        tp=np.sum((labels == cls) & (predictions == cls)); fp=np.sum((labels != cls) & (predictions == cls)); fn=np.sum((labels == cls) & (predictions != cls))
        precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.; f1.append(2*precision*recall/(precision+recall) if precision+recall else 0.)
    return {'accuracy': float(np.mean(labels == predictions)), 'macro_f1': float(np.mean(f1)), 'emergency_recall': float(np.mean(predictions[labels == 2] == 2)), 'distance_mae': float(np.mean(np.abs(distance_predictions-distance_truth)))}
