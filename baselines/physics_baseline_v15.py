"""V1.5 Physics Baseline：只使用带误差的当前观测，不读取隐藏真实动力学。"""
import math
G=9.81
def predict(row):
    speed=float(row['ego_speed']); gap=float(row['distance']); rel=float(row['relative_speed']); mu=float(row['road_friction']); slope=float(row['road_slope']); reaction=float(row['reaction_time'])
    decel=max(.8,mu*G*math.cos(slope)-G*math.sin(slope)); distance=speed*reaction+speed*speed/(2*decel); ttc=gap/rel if rel>.1 else 99.; margin=gap-distance
    risk='Emergency' if margin<0 or ttc<1.5 else ('Warning' if margin<12 or ttc<3.5 else 'Safe')
    return risk,distance
