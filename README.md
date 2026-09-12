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

## V2.1：移除 proxy leakage 的因果基线

> **V2 历史结果作废。** 原 V2 的 `brake_state` 由 `risk_level != Safe` 直接生成，`ego_acceleration` 又由该状态驱动，且 TTC 由隐藏真实状态计算；因此旧 V2 的高指标不能代表真实泛化能力。

V2.1 不修改 V1.5 Ground Truth Simulator 的标签逻辑，而是在独立生成器中重建观测—控制链路：控制器只读取延迟、带噪的当前或过去速度与距离观测；`ttc` 由这些观测重算；制动命令经过驾驶员反应延迟、执行器延迟和制动力建立过程后才影响车辆加速度。主 MLP 和 GRU 都**不输入 `brake_state`**。所有切分仍按 `scenario_id` 固定为 70/15/15，scaler 仅在 train 拟合，validation 用于 early stopping，test 只用于最终评估。

V2.1 数据审计报告在 `results/v21/data_audit_primary.json`：210/45/45 个场景，scenario overlap 为 0；禁用字段没有进入输入；TTC 可由保存的观测字段重算，`sensor_delay_ms` 与记录的 `effective_delay_ms` 完全一致（0/200/400 ms）；8 分位单特征 sanity check 没有发现验证准确率超过 95% 的代理变量。

| Model | Accuracy | Macro F1 | Emergency Recall | Distance MAE | Distance RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Physics Baseline | 60.06% | 52.62% | 84.44% | 9.390 m | 15.047 m |
| MLP（主实验，无 brake_state） | 89.70% | 85.06% | 91.53% | 3.907 m | 7.321 m |
| GRU（主实验，无 brake_state） | 91.39% | 87.03% | 89.82% | 3.227 m | 5.995 m |
| MLP 消融：无 ego_acceleration | 88.75% | 84.41% | 85.74% | 4.450 m | 8.190 m |
| MLP 消融：两者均无 | 87.06% | 82.10% | 88.78% | 4.417 m | 7.986 m |

README 主表采用 **aligned endpoint**：Physics、MLP、GRU 都只在 GRU 可评估的同一 3,195 个序列端点比较。完整 3,600 帧结果（含消融）保留在 `results/v21/all_frame_comparison.csv`；主表的对齐结果在 `results/v21/aligned_endpoint_comparison.csv`。GRU 用 10 个历史步（2 秒）。完整固定测试结果、预测、混淆矩阵和训练历史在 `results/v21/`。checkpoint 不提交 Git，训练后会产生于 `checkpoints/best_<model>_v21_<feature_set>.pt`。

```powershell
python data/generate_v21_dataset.py --scenarios 300 --steps 80 --seed 2026
python -c "from utils.split import write_splits; print(write_splits('data/synthetic_driving_v21.csv', 'data/v21_splits', 42))"
python -m training.train_v21 --model mlp --feature-set without_brake_state
python -m training.train_v21 --model gru --feature-set without_brake_state
python -m training.evaluate_v21_physics
python -m training.evaluate_v21 --model mlp --feature-set without_brake_state
python -m training.evaluate_v21 --model gru --feature-set without_brake_state
python -m training.report_v21
```

## V3：因果时序模型与公平端点比较

V3 严格复用 V2.1 的生成数据、标签、scenario-level train/validation/test split、leakage audit 与仅由 train 拟合的 scaler；主实验仍为 `without_brake_state`。新增模型全部以过去和当前的 10 个时间步（2 秒）预测当前端点，且每段序列按 `scenario_id` 独立构建，绝不跨场景。

- **Causal TCN**：48 通道、3 个 dilation 为 1/2/4 的 left-padded causal residual Conv1D block；卷积在时刻 *t* 无法读取 *t* 之后的帧。
- **Lightweight Transformer**：48 维输入投影与正弦位置编码、2 层/4 heads Transformer Encoder，并施加上三角 causal mask；仅取末端 token 进入共享分类与制动距离回归头。
- 所有神经模型使用 endpoint 训练标签计算 class weights、weighted CrossEntropy + 0.05×SmoothL1、AdamW、固定 seed，并只以 validation early stopping 选择 checkpoint。5/10/20 步只可作为 validation ablation；主测试固定为 10 步，不能据 test 调参。

V3 的最终表以 GRU 的 10-step test endpoints 为唯一索引，将 Physics、MLP、GRU、TCN、Transformer 重新过滤到同一端点；因此它不会把 MLP/Physics 的全部帧结果误与时序模型相比。训练与评估会同时记录 checkpoint、history、预测、混淆矩阵、参数量，以及在记录的运行环境中 warm-up 后同步测量的 batch-1 / batch-256 推理延迟。

```powershell
py -m training.train_v3 --model tcn --sequence-length 10
py -m training.evaluate_v3 --model tcn --sequence-length 10
py -m training.train_v3 --model transformer --sequence-length 10
py -m training.evaluate_v3 --model transformer --sequence-length 10
py -m training.benchmark_v3
py -m training.report_v3
```

若要完成要求的 validation-only 长度消融，分别训练 5、10、20 steps（不执行对应 test evaluation），然后汇总：

```powershell
py -m training.train_v3 --model tcn --sequence-length 5
py -m training.train_v3 --model tcn --sequence-length 20
py -m training.report_v3_ablation --model tcn
py -m training.train_v3 --model transformer --sequence-length 5
py -m training.train_v3 --model transformer --sequence-length 20
py -m training.report_v3_ablation --model transformer
```




### V3 最终结果

主表严格对齐到 45 个 held-out 场景的 3,195 个 GRU 可评估端点；完整机器可读结果在 `results/v3/aligned_endpoint_comparison.csv`。

| Model | Accuracy | Macro F1 | Emergency Recall | Distance MAE | Params | batch-1 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Physics | 60.06% | 52.62% | 84.44% | 9.390 m | 0 | 0.001 ms (CPU scalar) |
| MLP | 89.70% | 85.06% | 91.53% | 3.907 m | 2,980 | 0.055 ms (CPU) |
| GRU | 91.39% | 87.03% | 89.82% | 3.227 m | 16,996 | 0.458 ms (CPU) |
| Causal TCN | 86.01% | 80.36% | 86.04% | 2.997 m | 44,036 | 1.569 ms (CPU) |
| Lightweight Transformer | 88.39% | 83.47% | 89.59% | 3.395 m | 40,196 | 1.040 ms (CPU) |

TCN 的距离 MAE 最低，GRU 保持最高分类指标。V3 模型均不读取未来帧、`brake_state`、标签字段，且不会跨 `scenario_id` 形成窗口。

完整端点对齐结果与每模型 batch-1/batch-256 latency 位于 `results/v3/aligned_endpoint_comparison.csv`；运行环境（GPU 名称、PyTorch、CUDA runtime 和计时协议）位于 `results/v3/deployment_metrics.json`。本次基准为 PyTorch 2.14.0+cpu，CUDA 不可用、GPU 名称为空；若在 CUDA 主机复跑，文件会保存具体 GPU 名称和 runtime。Physics 是 CPU 上的 Python 标量基线，其延迟不应与 GPU 神经网络直接横比。

5/10/20-step 的轻量消融只读取 validation history，不调用 test：TCN 的最小 validation loss 分别为 0.421 / 0.458 / 0.442，Transformer 为 0.482 / 0.429 / 0.424。原始汇总保存在 `results/v3/tcn_validation_length_ablation.csv` 和 `results/v3/transformer_validation_length_ablation.csv`；主测试仍固定使用预先指定的 10-step checkpoint。

## V4：鲁棒性、可解释性与交互展示

V4 不再追逐 test 指标。它冻结 V2.1/V3 checkpoint，在同一批 3,195 个 held-out endpoints 上做描述性压力测试与遮挡解释；不会训练、调参或以这些结果选择模型。

```text
Noisy observations → train-only scaler → 10-step causal window
                                      ├→ GRU: classification reference
                                      ├→ TCN: stopping-distance reference
                                      └→ Transformer: lightweight temporal comparator
                                                     ↓
                       robustness / masking explanations / static scenario demo
```

压力测试覆盖确定性传感器噪声、额外一帧观测延迟、末端 distance/relative-speed/TTC dropout（均值插补）与低附着扰动。结果位于 `results/v4/robustness_stress_test.csv`：所有模型在轻微噪声与额外延迟下仅小幅下降，而 dropout 对 Transformer 的影响最大；低附着主要恶化停车距离 MAE，反映了冻结模型在分布外制动条件下的局限。这里的 `low_friction` 是**固定标签不变**的输入/协变量扰动：仅将模型窗口中的观测 `road_friction` 下调 0.25，不重新运行仿真，也不把它解释成“低附着导致的反事实 ground truth”。

可解释性使用“遮挡后 Macro F1 下降”而非梯度归因。`results/v4/feature_masking.csv` 显示 distance 与 relative_speed 是两类时序模型的主要分类依据；`results/v4/temporal_masking_heatmap.png` 与 `timestep_masking.csv` 显示最近时刻最重要，尤其是 Transformer 的当前端点。这些是模型行为诊断，并不证明因果关系。

### 一键复现 V4

```powershell
python -m training.robustness_v4
python -m training.interpret_v4
python demo/build_assets.py
python -m unittest tests.test_v4_guards
```

`tests/test_v4_guards.py` 守护 scenario split、train-only scaler、禁止字段、因果卷积的未来不可见性以及 sequence 不跨 scenario；GitHub Actions 在 PR 上运行相同 smoke test。交互 Demo 位于 `demo/index.html`：在该目录启动任意静态服务器后，可选择 held-out scenario、模型与端点，查看速度/间距/TTC、风险和真实/预测停车距离时间线。
