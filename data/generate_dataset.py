"""生成合成车辆纵向跟车与制动场景，不依赖外部数据或第三方包。"""
import argparse
import csv
import math
import random
from pathlib import Path

G = 9.81
FIELDS = ['scenario_id','time_s','scenario_type','ego_speed','lead_speed','distance','relative_speed','ttc','ego_acceleration','road_friction','road_slope','vehicle_mass','reaction_time','brake_state','risk_level','stopping_distance']

def risk_label(distance, ttc, stopping_distance):
    """以可用间距、TTC 和预测停车距离生成工程启发式风险标签。"""
    margin = distance - stopping_distance
    if margin < 0 or ttc < 1.5: return 'Emergency'
    if margin < 12 or ttc < 3.5: return 'Warning'
    return 'Safe'

def scenario_kind(index):
    return ['stable_following','gradual_lead_deceleration','sudden_lead_braking','high_speed_approach'][index % 4]

def generate_scenario(index, steps, dt, rng):
    kind = scenario_kind(index)
    wet = index % 5 == 0
    friction = rng.uniform(0.35, 0.55) if wet else rng.uniform(0.70, 1.00)
    slope = rng.uniform(-0.06, 0.06)  # 正值表示下坡，降低可用制动减速度。
    mass = rng.uniform(1100, 2400)
    reaction = rng.uniform(0.4, 1.8)
    ego = rng.uniform(18, 30) if kind != 'high_speed_approach' else rng.uniform(28, 38)
    lead = ego + rng.uniform(-2, 2) if kind == 'stable_following' else ego - rng.uniform(3, 12)
    distance = rng.uniform(30, 65) if kind != 'high_speed_approach' else rng.uniform(35, 75)
    rows = []
    for step in range(steps):
        time_s = step * dt
        lead_acc = 0.0
        if kind == 'gradual_lead_deceleration' and time_s > 4: lead_acc = -rng.uniform(0.4, 1.2)
        if kind == 'sudden_lead_braking' and 3 < time_s < 5.5: lead_acc = -rng.uniform(4.5, 7.5)
        if kind == 'high_speed_approach': lead_acc = -rng.uniform(0.0, 0.4)
        lead = max(0, lead + lead_acc * dt)
        relative_speed = ego - lead  # 正值代表自车正在接近前车。
        ttc = distance / relative_speed if relative_speed > 0.1 else 99.0
        effective_decel = max(0.8, friction * G * math.cos(slope) - G * math.sin(slope))
        stopping_distance = ego * reaction + ego * ego / (2 * effective_decel)
        brake = ttc < 3.5 or distance < stopping_distance + 8
        ego_acc = -min(effective_decel * 0.75, 6.5) if brake else rng.uniform(-0.25, 0.25)
        # 低幅噪声模拟传感器测量误差，但不会破坏真实连续状态。
        measured_distance = max(1, distance + rng.gauss(0, 0.25))
        measured_ego = max(0, ego + rng.gauss(0, 0.08))
        measured_lead = max(0, lead + rng.gauss(0, 0.08))
        risk = risk_label(measured_distance, ttc, stopping_distance)
        rows.append({'scenario_id': f'S{index:04d}', 'time_s': round(time_s, 2), 'scenario_type': kind, 'ego_speed': round(measured_ego, 3), 'lead_speed': round(measured_lead, 3), 'distance': round(measured_distance, 3), 'relative_speed': round(measured_ego-measured_lead, 3), 'ttc': round(ttc, 3), 'ego_acceleration': round(ego_acc, 3), 'road_friction': round(friction, 3), 'road_slope': round(slope, 4), 'vehicle_mass': round(mass, 1), 'reaction_time': round(reaction, 3), 'brake_state': int(brake), 'risk_level': risk, 'stopping_distance': round(stopping_distance, 3)})
        distance = max(1, distance + (lead - ego) * dt)
        ego = max(0, ego + ego_acc * dt)
    return rows

def main():
    p = argparse.ArgumentParser(); p.add_argument('--scenarios', type=int, default=150); p.add_argument('--steps', type=int, default=80); p.add_argument('--seed', type=int, default=42); p.add_argument('--output', default='data/synthetic_driving.csv'); a = p.parse_args()
    rng = random.Random(a.seed); output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS); writer.writeheader()
        for index in range(a.scenarios): writer.writerows(generate_scenario(index, a.steps, 0.2, rng))
    print(f'Generated {a.scenarios} scenarios, {a.scenarios*a.steps} time steps: {output}')
if __name__ == '__main__': main()
