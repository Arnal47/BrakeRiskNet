# BrakeRiskNet V1

面向自动驾驶纵向安全的车辆制动风险预测与停车距离估计项目。V1 生成连续跟车/制动场景，并以可解释的 Physics Baseline 作为后续 PyTorch 时序模型的比较起点。

## 数据字段

每一行是一个时间步，`scenario_id` 标识完整驾驶场景。主要输入包括 `ego_speed`、`lead_speed`、`distance`、`relative_speed`、`ttc`、`ego_acceleration`、`road_friction`、`road_slope`、`vehicle_mass`、`reaction_time`、`brake_state`；监督目标为 `risk_level`（Safe / Warning / Emergency）和 `stopping_distance`。

## 物理基线

预测停车距离：`d = v × reaction_time + v² / (2a)`，其中 `a = μg cos(slope) − g sin(slope)`。正坡度按下坡定义，因此降低有效减速度。风险由停车余量和 TTC 共同判断。

## 严格数据划分

绝不按单帧随机切分。脚本按 `scenario_id` 将完整连续场景分配至 train / validation / test（70% / 15% / 15%），并在训练前检查三者 scenario 交集为 0。

## 运行 smoke test

```powershell
python data/generate_dataset.py --scenarios 150 --steps 80
python utils/split.py
python evaluate_baseline.py
```

结果将写入 `results/`。V1 不训练神经网络，不安装 CUDA Toolkit；下一阶段可在同一 group-aware 划分上加入 PyTorch LSTM/GRU/Transformer。

## V1.5：真实动力学标签与泄漏修复

V1.5 保留 `scenario_id` 的 group-aware 70/15/15 切分，但将 Ground Truth Simulator 与 Physics Baseline 彻底分离。真实标签来自包含执行器延迟、制动力建立、效率差异、时变路面附着、坡度、测量延迟、噪声和 dropout 的逐步车辆动力学；Physics Baseline 只读取带误差的当前观测。详见 [V15_AUDIT.md](V15_AUDIT.md)。

```powershell
python data/generate_v15_dataset.py --scenarios 300 --steps 80
python -c "from utils.split import write_splits; print(write_splits('data/synthetic_driving_v15.csv', 'data/v15_splits', 42))"
python evaluate_v15_baseline.py
```

## V2：无泄漏神经网络基线

V1 的 99% 指标不可信，因为标签直接沿用了 Physics Baseline 的公式和阈值。V1.5 将真实动力学标签与基线分离；V2 在固定的 scenario_id 级 70/15/15 split 上训练 MLP 与 GRU。数据审计结果见 `results/v2/data_audit.json`：三集合 scenario overlap 为 0，输入不包含 `scenario_id`、`risk_level`、`stopping_distance`，并且标准化器只用 train 拟合。

网络只输入当前 noisy observable state：ego/lead speed、distance、relative speed、TTC、ego acceleration、摩擦/坡度/质量/反应时间估计、sensor delay 和 brake state。MLP 使用 `12 -> 64 -> 32` shared backbone；GRU 使用 10 个历史时间步（2 秒）输入、64 维 GRU hidden state、32 维 shared head；两者都具有风险分类头和停车距离回归头。

| Model | Accuracy | Macro F1 | Emergency Recall | Distance MAE | Distance RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Physics Baseline | 26.14% | 21.65% | 78.57% | 12.436 m | 17.410 m |
| MLP | 96.53% | 81.43% | 65.00% | 5.190 m | 9.383 m |
| GRU | 99.62% | 95.12% | 83.33% | 4.569 m | 8.750 m |

三者使用同一批 45 个 test 场景。MLP 与 Physics 评估 3,600 个时间步；GRU 由于 10 步历史窗口只评估每场景第 10 步后的 3,195 个合法序列末端。最终 checkpoint：`checkpoints/best_mlp_v2.pt` 与 `checkpoints/best_gru_v2.pt`。完整指标、预测、混淆矩阵、loss 曲线和对比表位于 `results/v2/`。

运行：`python -m training.train_v2 --model mlp`、`python -m training.evaluate_v2 --model mlp`、`python -m training.train_v2 --model gru --sequence-length 10`、`python -m training.evaluate_v2 --model gru`。

## 本地 checkpoint 与复现

为保持 Git 仓库轻量，训练权重不会提交：`checkpoints/best_mlp_v2.pt` 与 `checkpoints/best_gru_v2.pt` 均由 `.gitignore` 排除。克隆后，请先运行 V1.5 生成与场景级切分命令，再运行对应的 V2 训练命令；最佳 checkpoint 会在相同路径自动生成。之后可运行 `python -m training.evaluate_v2 --model mlp` 或 `python -m training.evaluate_v2 --model gru`，所有测试结果始终使用固定的 V1.5 test split。
