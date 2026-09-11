"""V2.1：真值标签与因果观测/控制链路分离。"""
import argparse, csv, math, random
from pathlib import Path
try:
    from data.generate_v15_dataset import G, FIELDS, true_stop_distance, gt_risk
except ModuleNotFoundError:
    from generate_v15_dataset import G, FIELDS, true_stop_distance, gt_risk
SAMPLE_MS = 200
FIELDS_V21 = FIELDS[:FIELDS.index('brake_state')] + ['effective_delay_ms'] + FIELDS[FIELDS.index('brake_state'):]

def make_scenario(index, steps, rng):
    kind=['stable_following','gradual_lead_deceleration','sudden_lead_braking','high_speed_approach'][index%4]
    wet=index%5==0; condition='wet_low_adhesion' if wet else 'dry'; true_mu=rng.uniform(.30,.52) if wet else rng.uniform(.68,.98)
    slope,mass=rng.uniform(-.06,.06),rng.uniform(1100,2400); reaction=rng.uniform(.45,1.8); actuator_delay,ramp,efficiency=rng.uniform(.08,.35),rng.uniform(.12,.45),rng.uniform(.72,.98)
    # 延迟只取采样周期整数倍；字段值与实际 buffer 索引完全一致。
    effective_delay_ms=rng.choice([0,SAMPLE_MS,2*SAMPLE_MS]); delay_steps=effective_delay_ms//SAMPLE_MS
    ego=rng.uniform(18,30) if kind!='high_speed_approach' else rng.uniform(28,38); lead=ego+rng.uniform(-2,2) if kind=='stable_following' else ego-rng.uniform(3,12); gap=rng.uniform(30,70)
    buffer=[];last=None;rows=[];command_time=None
    for step in range(steps):
        t=step*SAMPLE_MS/1000; lead_acc=0 if kind=='stable_following' else -.8 if kind=='gradual_lead_deceleration' and t>4 else -6.2 if kind=='sudden_lead_braking' and 3<t<5.5 else -.2 if kind=='high_speed_approach' else 0
        lead=max(0,lead+lead_acc*.2); raw={'ego_speed':ego+rng.gauss(0,.12),'lead_speed':lead+rng.gauss(0,.12),'distance':max(1,gap+rng.gauss(0,.35))};buffer.append(raw)
        observed=buffer[max(0,len(buffer)-1-delay_steps)]
        if rng.random()<.025 and last is not None: observed=last
        last=observed; obs_rel=observed['ego_speed']-observed['lead_speed']; obs_ttc=observed['distance']/obs_rel if obs_rel>.1 else 99.
        trigger_gap=max(7.,observed['ego_speed']*reaction*.65+4.)
        if command_time is None and (obs_ttc<3.0 or observed['distance']<trigger_gap): command_time=t
        elapsed=-1 if command_time is None else t-command_time-reaction; phase=max(0.,min(1.,(elapsed-actuator_delay)/ramp)) if elapsed>=0 else 0.
        mu_now=max(.18,true_mu*(1+.08*math.sin(.7*t))); ego_acc=-efficiency*phase*mu_now*G*math.cos(slope)+G*math.sin(slope)
        if phase==0: ego_acc=rng.uniform(-.18,.12)+G*math.sin(slope)*.03
        stop,_=true_stop_distance(ego,true_mu,slope,actuator_delay,ramp,efficiency); risk=gt_risk(gap,ego,lead,kind,true_mu,slope,actuator_delay,ramp,efficiency)
        rows.append({'scenario_id':f'S{index:04d}','time_s':round(t,2),'scenario_type':kind,'road_condition':condition,'ego_speed':round(observed['ego_speed'],3),'lead_speed':round(observed['lead_speed'],3),'distance':round(observed['distance'],3),'relative_speed':round(obs_rel,3),'ttc':round(obs_ttc,3),'ego_acceleration':round(ego_acc,3),'road_friction':round(max(.2,true_mu+rng.gauss(0,.07)),3),'road_slope':round(slope+rng.gauss(0,.008),4),'vehicle_mass':round(mass+rng.gauss(0,80),1),'reaction_time':round(max(.2,reaction+rng.gauss(0,.25)),3),'sensor_delay_ms':effective_delay_ms,'effective_delay_ms':effective_delay_ms,'brake_state':int(phase>0),'risk_level':risk,'stopping_distance':round(stop,3)})
        gap=max(1,gap+(lead-ego)*.2);ego=max(0,ego+ego_acc*.2)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenarios',type=int,default=300);p.add_argument('--steps',type=int,default=80);p.add_argument('--seed',type=int,default=2026);p.add_argument('--output',default='data/synthetic_driving_v21.csv');a=p.parse_args();rng=random.Random(a.seed);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS_V21);w.writeheader()
        for i in range(a.scenarios):w.writerows(make_scenario(i,a.steps,rng))
    print(f'V2.1 generated {a.scenarios} scenarios, {a.scenarios*a.steps} time steps: {out}')
if __name__=='__main__':main()
