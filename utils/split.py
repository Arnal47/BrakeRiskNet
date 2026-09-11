"""按 scenario_id 的 group-aware 数据切分，禁止任意场景泄漏。"""
import csv
import random
from collections import defaultdict
from pathlib import Path

def assert_no_scenario_leakage(splits):
    groups = {name: {row['scenario_id'] for row in rows} for name, rows in splits.items()}
    overlaps = (groups['train'] & groups['validation']) | (groups['train'] & groups['test']) | (groups['validation'] & groups['test'])
    if overlaps: raise RuntimeError(f'Scenario leakage detected: {sorted(overlaps)[:5]}')
    return 0

def split_by_scenario(rows, seed=42, ratios=(0.70, 0.15, 0.15)):
    grouped = defaultdict(list)
    for row in rows: grouped[row['scenario_id']].append(row)
    ids = sorted(grouped); rng = random.Random(seed); rng.shuffle(ids)
    n = len(ids); n_train, n_val = round(n*ratios[0]), round(n*ratios[1])
    ids_by_split = {'train': ids[:n_train], 'validation': ids[n_train:n_train+n_val], 'test': ids[n_train+n_val:]}
    splits = {name: [row for scenario_id in selected for row in grouped[scenario_id]] for name, selected in ids_by_split.items()}
    assert_no_scenario_leakage(splits)
    return splits, {name: len(selected) for name, selected in ids_by_split.items()}

def write_splits(input_path='data/synthetic_driving.csv', output_dir='data/splits', seed=42):
    with open(input_path, newline='', encoding='utf-8') as file: rows = list(csv.DictReader(file))
    splits, counts = split_by_scenario(rows, seed=seed); output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        with (output/f'{name}.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(part)
    return counts, assert_no_scenario_leakage(splits)

if __name__ == '__main__':
    counts, overlap = write_splits(); print(f'Scenario split: {counts}, overlap={overlap}')
