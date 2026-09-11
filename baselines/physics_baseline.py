"""反应距离 + 制动距离的可解释物理基线。"""
import math
G = 9.81

def predict(row):
    """计算预估停车距离和风险等级。

    d_stop = v*t_reaction + v²/(2*a), a=μg cos(slope)-g sin(slope)。
    此处 slope>0 表示下坡，会降低有效制动减速度。
    """
    speed, distance = float(row['ego_speed']), float(row['distance'])
    friction, slope, reaction = float(row['road_friction']), float(row['road_slope']), float(row['reaction_time'])
    relative_speed = float(row['relative_speed'])
    effective_decel = max(0.8, friction*G*math.cos(slope) - G*math.sin(slope))
    stopping_distance = speed*reaction + speed*speed/(2*effective_decel)
    ttc = distance/relative_speed if relative_speed > .1 else 99.
    margin = distance-stopping_distance
    risk = 'Emergency' if margin < 0 or ttc < 1.5 else ('Warning' if margin < 12 or ttc < 3.5 else 'Safe')
    return risk, stopping_distance
