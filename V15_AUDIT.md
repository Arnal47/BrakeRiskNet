# BrakeRiskNet V1.5 数据与标签泄漏审计

## V1 的问题

V1 将 `risk_level` 与 `stopping_distance` 直接按 Physics Baseline 的反应距离、制动距离和阈值生成。因此同一组可观测字段与几乎同一套公式同时定义标签、又被基线读取，形成“公式预测公式”，导致 test accuracy 99.29%、MAE 0.211 m 的虚高结果。

## V1.5 的修复

- `Ground Truth Simulator` 在 `data/generate_v15_dataset.py` 中逐时间步积分真实制动过程；标签不会调用 `baselines/physics_baseline_v15.py`。
- 真实停车距离来自实际停车位置，仿真包含执行器延迟、制动力建立时间、效率差异、时变附着、坡度与质量。
- `risk_level` 由未来 5 秒内的最小间距和真实碰撞结果确定，而不是复制基线阈值。
- 基线只观察带噪声、延迟、2.5% dropout 的速度/距离，以及带估计误差的摩擦、坡度、质量和反应时间；真实参数保持隐藏。
- 切分始终按 `scenario_id`，70/15/15，smoke test 的三集合 overlap 为 0。

## Smoke test（300 场景）

| 指标 | V1 | V1.5 |
| --- | ---: | ---: |
| risk accuracy | 99.29% | 26.14% |
| Macro F1 | 99.24% | 21.65% |
| stopping distance MAE | 0.211 m | 12.436 m |
| stopping distance RMSE | 0.298 m | 17.410 m |

V1.5 数值下降是预期的：它表明 Physics Baseline 无法完全观测或精确模拟真实生成机制。高速接近是当前最困难切片（accuracy 0），湿滑路面 accuracy 为 28.13%。类别分布为 Safe 20,054、Warning 2,928、Emergency 1,018，后续时序模型应使用类别权重、重采样或合适的风险敏感指标。
