# V2.1 proxy leakage audit

## Why V2 is invalid

V2 generated `brake_state` with `risk_level != 'Safe'`, then generated `ego_acceleration` from that state. It also stored TTC from hidden true gap and true relative speed. These are label-derived or hidden-state proxies, so the V2 metrics are not valid experimental results.

## V2.1 controls

- V1.5 Ground Truth Simulator remains unchanged and continues to own labels.
- The V2.1 controller receives only delayed/noisy observed ego speed, lead speed and distance. It does not read `risk_level`, stopping distance, future minimum gap or collision outcomes.
- Observed TTC is recomputed from delayed/noisy observed distance and relative speed.
- Brake command uses current/past observation, reaction delay, actuator delay and actuator ramp. Vehicle acceleration follows this causal command and dynamics.
- Main MLP and GRU exclude `brake_state`. The supplied ablations assess removal of acceleration and both fields.
- `scenario_id` group split remains 70/15/15, with zero overlap. The scaler is fitted on train only; validation selects the checkpoint and held-out test is final-only.

## Result interpretation

V2.1 is the authoritative neural-baseline result. Its lower metrics are expected after removing proxy shortcuts. See `results/v21/model_comparison.csv` and `results/v21/data_audit.json`.
