"""Build static demo data from committed prediction artefacts; no model inference in browser."""
import csv, json
from pathlib import Path

ROOT=Path(__file__).parent; REPO=ROOT.parent
SOURCES={
    'Physics':REPO/'results/v21/physics_test_predictions.csv',
    'MLP':REPO/'results/v21/mlp_without_brake_state_test_predictions.csv',
    'GRU':REPO/'results/v21/gru_without_brake_state_test_predictions.csv',
    'TCN':REPO/'results/v3/tcn_without_brake_state_len10_test_predictions.csv',
    'Transformer':REPO/'results/v3/transformer_without_brake_state_len10_test_predictions.csv',
}


def read(path):
    with open(path,newline='',encoding='utf-8') as handle:return list(csv.DictReader(handle))


def main():
    predictions={name:{(row['scenario_id'],row['time_s']):row for row in read(path)} for name,path in SOURCES.items()}
    common=set(predictions['GRU'])
    rows=read(REPO/'data/v21_splits/test.csv'); scenarios={}
    for row in rows:
        key=(row['scenario_id'],row['time_s'])
        if key not in common:continue
        item={'time_s':float(row['time_s']),'ego_speed':float(row['ego_speed']),'lead_speed':float(row['lead_speed']),'distance':float(row['distance']),'ttc':float(row['ttc']),'true_risk':row['risk_level'],'true_distance':float(row['stopping_distance']),'models':{}}
        for name,values in predictions.items():
            prediction=values[key];item['models'][name]={'risk':prediction['predicted_risk'],'distance':float(prediction['predicted_distance'])}
        scenarios.setdefault(row['scenario_id'],[]).append(item)
    (ROOT/'assets').mkdir(exist_ok=True)
    (ROOT/'assets/scenario_data.json').write_text(json.dumps(scenarios,separators=(',',':')),encoding='utf-8')
    print({'scenarios':len(scenarios),'endpoints':sum(len(value) for value in scenarios.values())})


if __name__=='__main__':main()
