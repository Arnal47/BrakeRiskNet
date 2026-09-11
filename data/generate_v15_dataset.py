"""V1.5 Ground Truth Simulator：用隐藏真实动力学生成标签，绝不调用 baseline。"""
import argparse, csv, math, random
from pathlib import Path
G=9.81
FIELDS=['scenario_id','time_s','scenario_type','road_condition','ego_speed','lead_speed','distance','relative_speed','ttc','ego_acceleration','road_friction','road_slope','vehicle_mass','reaction_time','sensor_delay_ms','brake_state','risk_level','stopping_distance']

def true_stop_distance(v, mu, slope, actuator_delay, ramp_time, efficiency, dt=.02):
    """真实制动：响应延迟、制动力建立、随时间变化的附着和坡度逐步积分至停车。"""
    speed, distance, t= v, 0., 0.
    while speed>0.01 and t<30:
        phase=max(0.,min(1.,(t-actuator_delay)/ramp_time))
        mu_t=max(.18,mu*(1+.08*math.sin(.7*t)))
        decel=max(.15, efficiency*phase*mu_t*G*math.cos(slope)-G*math.sin(slope))
        distance+=speed*dt; speed=max(0.,speed-decel*dt); t+=dt
    return distance,t

def gt_risk(distance, ego, lead, kind, mu, slope, delay, ramp, efficiency):
    """观察未来 5 秒真实演化：最小间距、碰撞与实际制动过程共同定义标签。"""
    min_gap, collision=distance,False
    for i in range(50):
        t=i*.1; lead_decel=0
        if kind=='sudden_lead_braking' and t<2.5: lead_decel=-6.5
        elif kind=='gradual_lead_deceleration': lead_decel=-.9
        lead=max(0,lead+lead_decel*.1)
        stop,_=true_stop_distance(ego,mu,slope,delay,ramp,efficiency)
        brake_decel=max(0, min(5.5,(ego/ max(stop,1))*2)) if distance<stop+10 else 0
        ego=max(0,ego-brake_decel*.1); distance+= (lead-ego)*.1; min_gap=min(min_gap,distance)
        collision|=distance<=0
    ttc=distance/(ego-lead) if ego>lead+.1 else 99
    return 'Emergency' if collision or min_gap<2 else ('Warning' if min_gap<12 or ttc<3.5 else 'Safe')

def make_scenario(index,steps,rng):
    kind=['stable_following','gradual_lead_deceleration','sudden_lead_braking','high_speed_approach'][index%4]
    wet=index%5==0; condition='wet_low_adhesion' if wet else 'dry'
    true_mu=rng.uniform(.30,.52) if wet else rng.uniform(.68,.98); slope=rng.uniform(-.06,.06); mass=rng.uniform(1100,2400)
    true_reaction=rng.uniform(.45,1.8); actuator_delay=rng.uniform(.08,.35); ramp=rng.uniform(.12,.45); efficiency=rng.uniform(.72,.98); sensor_delay=rng.choice([0,100,200,300])
    ego=rng.uniform(18,30) if kind!='high_speed_approach' else rng.uniform(28,38); lead=ego+rng.uniform(-2,2) if kind=='stable_following' else ego-rng.uniform(3,12); gap=rng.uniform(30,70)
    buffer=[]; last=None; rows=[]
    for step in range(steps):
        t=step*.2; lead_acc=0 if kind=='stable_following' else (-.8 if kind=='gradual_lead_deceleration' and t>4 else (-6.2 if kind=='sudden_lead_braking' and 3<t<5.5 else -.2 if kind=='high_speed_approach' else 0))
        lead=max(0,lead+lead_acc*.2); rel=ego-lead; ttc=gap/rel if rel>.1 else 99.; stop,_=true_stop_distance(ego,true_mu,slope,actuator_delay,ramp,efficiency)
        risk=gt_risk(gap,ego,lead,kind,true_mu,slope,actuator_delay,ramp,efficiency)
        brake=risk!='Safe'; ego_acc=-min(5.5,true_mu*G*efficiency) if brake else rng.uniform(-.2,.2)
        raw={'ego_speed':ego+rng.gauss(0,.12),'lead_speed':lead+rng.gauss(0,.12),'distance':max(1,gap+rng.gauss(0,.35))}
        buffer.append(raw); delay_steps=sensor_delay//200; observed=buffer[max(0,len(buffer)-1-delay_steps)]
        if rng.random()<.025 and last: observed=last
        last=observed
        # 基线只得到有误差的参数估计，真实参数完全隐藏在 simulator 内。
        row={'scenario_id':f'S{index:04d}','time_s':round(t,2),'scenario_type':kind,'road_condition':condition,'ego_speed':round(observed['ego_speed'],3),'lead_speed':round(observed['lead_speed'],3),'distance':round(observed['distance'],3),'relative_speed':round(observed['ego_speed']-observed['lead_speed'],3),'ttc':round(ttc,3),'ego_acceleration':round(ego_acc,3),'road_friction':round(max(.2,true_mu+rng.gauss(0,.07)),3),'road_slope':round(slope+rng.gauss(0,.008),4),'vehicle_mass':round(mass+rng.gauss(0,80),1),'reaction_time':round(max(.2,true_reaction+rng.gauss(0,.25)),3),'sensor_delay_ms':sensor_delay,'brake_state':int(brake),'risk_level':risk,'stopping_distance':round(stop,3)}
        rows.append(row); gap=max(1,gap+(lead-ego)*.2); ego=max(0,ego+ego_acc*.2)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scenarios',type=int,default=300); p.add_argument('--steps',type=int,default=80); p.add_argument('--seed',type=int,default=2026); p.add_argument('--output',default='data/synthetic_driving_v15.csv'); a=p.parse_args(); rng=random.Random(a.seed); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); [w.writerows(make_scenario(i,a.steps,rng)) for i in range(a.scenarios)]
    print(f'V1.5 generated {a.scenarios} scenarios, {a.scenarios*a.steps} time steps: {out}')
if __name__=='__main__':main()
