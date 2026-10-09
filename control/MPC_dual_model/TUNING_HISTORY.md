# MPC Tuning History

This file records tested parameter sets that may need to be restored exactly.

## 2026-08-15: offline horizon/forward-force balance candidate

The two latest actuator-model traces were replayed with the confirmed visual
misidentification interval removed from
`experimental_auto_20260815_201912.jsonl` (`frame 22459--22516`).  Increasing
forward position weight or reducing forward absolute-force cost alone changed
the predicted terminal forward P95 by only a few millimetres while increasing
the first force command.  A longer horizon provided the useful part of the
dynamic preview, but an unweighted `horizon=15` raised force-rate activity too
much.

The current experimental candidate therefore uses:

```text
horizon = 15
position_weights = [800, 350, 900]
velocity_weights = [150, 150, 200]
force_weights = [0.60, 0.50, 0.80]
delta_force_weights = [5.0, 0.5, 3.0]
```

The change is upper-computer only.  It has passed the translation regression
suite and `EXPERIMENTAL AUTO READY`, but it has not yet been validated on the
vehicle.  No vision, lower-controller PID/mixing, `tau_h`, actuator model,
force limits, or final motor limits were changed.

## 2026-08-15: actuator delay queue completed offline, still disabled in AUTO

The optional upper-computer actuator model now retains the command-delay queue
between successive MPC solves.  Each solve re-anchors the current actuator
force to the measured `tau_achieved` while preserving commands that are still
inside the configured pure-delay queue.  The first-order actuator state and
`predicted_actuator_force_sequence` therefore no longer restart from the
latest achieved force on every solve.  `reset()` clears the queue.

The active experimental JSON remains unchanged:

```text
actuator_model_enabled = false
actuator_pure_delay_s = 0.08
actuator_time_constant_s = 0.15
```

These values remain offline candidates, not accepted real-device parameters.
Fixed-state replay of the two latest traces with the candidate enabled reduced
forward force-command P95 from approximately `4.40` to `4.12 N` in the short
retry and from `4.57` to `4.28 N` in the preceding long run.  Forward force
limit occupancy fell from about `4.3%` to `1.2%` and from `4.4%` to `2.3%`,
respectively.  This is not a closed-loop performance claim; the replay used
recorded states and forces.  No lower-controller firmware or vision code was
changed, and the model is not enabled for the next armed run until a separate
pool-top-camera comparison is completed.

## 2026-08-15: translation gated-EMA model-1 base restored

At the operator's request, only the translation `MPC_dual_model` tracker has
returned from the per-update base slew limiter to the earlier per-axis
`gated_ema` rule.  Eligible axes use `alpha=0.03` outside the strict steady
region and `alpha=0.01` inside it; eligibility remains `|position error| <=
0.20 m` and `|relative velocity| <= 0.08 m/s`, with strict steady thresholds
`0.03 m` and `0.02 m/s`.  All three FRD axes are enabled.

The independent Fossen term `tau_h=[0,0,0.807290] N`, model-2 equation,
absolute-total-force cost, actual-force rate reference, yaw-frame rotation,
camera constraints, and fallback to `tau_h` remain unchanged.  The
rotation-aware `MPC_dual_model_yaw` tracker is intentionally not changed in
this step.

## 2026-08-15: absolute-total-force cost restored for startup overshoot

The latest fusion run showed a large startup burst while `tau_base` and the
adaptive model-1 weight were still ramping from their initial values.  The QP
was previously centered by the fusion-weighted reference
`tau_eq=W*tau_base+(I-W)*tau_h`, which created a linear incentive to preserve a
large nonzero propulsion force.  The active controller now uses

```text
J_force = sum_j tau[j]^T R tau[j]
```

The dual-model state equations, fixed-per-solve `tau_base`, independent
`tau_h`, actual-force rate reference, force limits, and delta-force cost are
unchanged.  `MPCResult.force_reference` remains only as a compatibility trace
field and is now zero.  This is an upper-computer change; it does not require
lower-controller firmware or vision changes.  The next water run must compare
startup overshoot and stationary error before changing position or fusion
weights.

## Next physical session: firmware first, then a fixed-model-2 isolation run

The operator shut down the hardware after the 2026-08-15 offline fusion
validation.  Before the next physical test, compile and flash the lower
controller from `V4pro1_MPC`, then complete the normal disarmed/readiness
checks.  This is an explicit operator request even though the proposed fusion
experiment itself is an upper-computer change.

The next proposed isolation changes only the **forward** fusion selection to
fixed model 2 (`a1_forward=0`); right/down remain at their current adaptive
settings so a forward diagnosis is not confounded by simultaneous heave/sway
changes.  Do not change MPC costs, base slew, PID, vision, horizon, or limits
in the same run.  This candidate is recorded as a plan and is not yet active
in `finesub_v4pro1_mpc.json`.

The water sequence is: stationary target for at least 10 s, one slow forward
constant-speed pass, stationary stop for at least 10 s, then one reverse pass
and final stop.  Interpret the result by phase:

- If fixed model 2 still has poor stationary hold or low-frequency chasing,
  fusion/model-1 memory is not the primary cause; investigate actuator
  lag/deadband, variable vision timing, and then the forward MPC costs.
- If stationary hold becomes clearly stable but stopping with adaptive fusion
  was unstable, model-1 `tau_base` credibility/fusion is the primary cause.
- Constant-speed lag under fixed model 2 is structurally expected because it
  has no matched-speed operating-force feedforward.  That lag alone does not
  prove the current position/force weights are wrong; compare it with the
  stationary result and the earlier adaptive runs.

## 2026-08-15: reject longer fusion history as a standalone model-separation fix

The read-only validation tool `fusion_identifiability_validation.py` replayed
five translation-only traces against their independently recorded overhead
AprilTag bags.  Tag 17 supplied target motion and tags 15/16 supplied vehicle
motion.  The aggregate contained 856 stationary samples, 1,038 matched-motion
samples, and 481 stricter matched-motion samples for which both target and
vehicle planar acceleration were below `0.04 m/s2`.  No hardware transport was
opened and the active JSON was not changed.

Actually retaining completed five- and eight-step predictions materially
increased the forward candidate separation.  In strict steady cruise, median
separation rose from `0.78 mm` with the maintained effective two-step
staircase to `2.76/5.74 mm` at five/eight steps; P95 rose from `3.07 mm` to
`11.14/21.66 mm`.  Separation magnitude alone did not make the selection
correct: model 1 had lower scored MSE in only `40.1/39.3/41.2%` of strict
steady-cruise samples for effective horizons `2/5/8`.  Even after requiring
the logged base to change by less than `0.10 N/update` and differ from achieved
force by less than `0.30 N`, the corresponding fractions were only
`49.4/48.3/44.8%` over 174 samples.

The tested evidence-insufficient fallback decayed toward model 2 instead of
holding a stale model-1 weight.  With eight steps and a `1e-5 m2` threshold it
reduced stationary samples above model-1 weight `0.5` to `3.4%`, but classified
`72.4%` of matched-motion samples below `0.5`; its strict-cruise median weight
was approximately `0.01`.  Higher `1e-4 m2` thresholds collapsed even more
strongly to model 2.  Therefore neither longer fusion history nor a
model-2 ambiguity default is accepted as the next real-device change: the
combination mostly suppresses model 1 rather than identifying matched-speed
motion.  The report is
`calibration_logs/fusion_identifiability_validation_20260815.json`.

The result isolates a deeper mismatch in the identification signal.  The
logged applied-throttle force used as `tau_base` can still include correction,
actuator lag/deadband, and dynamic thrust-map error; the fixed `0.10 s`
propagation also accumulates error across real `0.10--0.20 s` vision intervals.
In addition, candidates are scored against a Kalman posterior whose prior
already uses the current fused model.  The next offline comparison must change
only the scoring/timing evidence—raw measurement versus a model-independent
posterior, then recorded variable-time propagation—before any fusion or MPC
parameter is changed in water.

## 2026-08-14: rate-limit the model-1 operating-force update

The operator observed that the model tracked constant-speed motion better
than a stationary target. In `experimental_auto_20260814_230123.jsonl`, after
target motion ended, five-second forward-error P95 repeatedly reached roughly
`20--54 cm`. During those chase cycles the adaptive forward model-1 weight
rose as high as approximately `0.62--0.82`, while the per-solve base copied
motion/correction force over approximately `-2.5--+1.4 N`. Increasing the
forward delta-force cost to `6.0` was rejected and restored to `4.0` because
it delayed unloading without preventing each achieved correction from
becoming the next model-1 equilibrium.

At the operator's request, model-1 base update and actual force-rate handling
are now separated. The persistent base obeys
`tau_base += clip(tau_achieved-tau_base, +/-base_slew_limit)` before each
solve, then remains fixed over the complete horizon. The first planned-force
difference still uses the unfiltered actual `tau_achieved`; therefore the
actuator slew constraint and feedback truth are unchanged. The first trial
used `[0.20,0.80,1.00] N/update`, so only forward was newly restricted, at a
nominal `2 N/s` under the fixed `0.10 s` model step. The old horizontal EMA is
not restored. Trace output records both the fixed limited base and its
difference from achieved force.

Frozen replay of the rejected trace reduced mean/P95 forward base change from
`0.368/0.800 N` to `0.176/0.200 N`; the resulting first-step force-change
bound fraction changed from `29.0%` to `26.2%`. This replay establishes the
implemented limit but cannot predict the new closed-loop stopping behavior.

The water run `experimental_auto_20260814_231713.jsonl` rejected the initial
`0.20 N/update` rate as too slow. The operator moved the fish during part of
the run, so those intervals are not labelled as stationary response. Across
the full 2700-update trace, however, the forward base step was at its `0.20 N`
limit in `75.8%` of adjacent updates, while `|tau_base-tau_achieved|` reached
`0.418/3.669/5.930 N` at P50/P95/max. This directly demonstrates persistent
base lag independent of target-motion labelling. The operator also observed
the slow update in real time. Only the forward base limit is therefore raised
to `0.40 N/update` (nominal `4 N/s`); right/down remain `0.80/1.00`, and all
weights, force-rate limits, fusion, PID, and vision settings remain unchanged.
Frozen replay of the same achieved-force sequence reduces forward
`|tau_base-tau_achieved|` P50/P95/max from `0.418/3.669/5.930 N` at `0.20`
to `0.000/2.376/4.829 N` at `0.40`; base updates at the selected limit fall
from `76.0%` to `39.0%`. This verifies the intended reduction in lag but does
not substitute for the next closed-loop water test.
The post-stop disarmed check recorded 59 fresh frames with state IDLE, armed
false, zero received channels and motor outputs, and no failsafe or rejection.

The next water run `experimental_auto_20260814_234007.jsonl` showed that
`0.40 N/update` was still too slow during repeated forward motion. Across
1032 control updates, the forward base step was at the configured limit in
`62.8%` of eligible updates and `|tau_base-tau_achieved|` reached
`0.298/2.355/3.315 N` at P50/P95/max. The forward model-1 weight median was
only `0.023`. The final collision interval also contained repeated visual
holds and large right error, so it is not used as clean forward-response
evidence. Frozen replay of the same achieved-force sequence, preserving
reacquisition resets, predicts that a forward limit of `0.60 N/update`
reduces base-lag P50/P95/max to `0.000/0.600/1.579 N`; `35.3%` of updates
reach the new limit. The `0.60 N/update` water run
`experimental_auto_20260814_234928.jsonl` then rejected that candidate. In
the final ten seconds with the target held stationary, forward error remained
about `-5 cm`, model-1 weight rose to a `0.973` median, and the base/achieved
force means reached approximately `-3.67/-3.62 N`. The forward command mean
was `-0.142` and hit the `-0.20` limit in `28%` of updates. The faster limiter
therefore allowed correction force to become a false matched-speed
equilibrium and did not restore static tracking. The active forward base
limit is reverted to `0.40 N/update` as the safer prior value; this rollback
does not claim that `0.40` solves the underlying observability and actuator-
model problem. All other parameters remain unchanged.

At the operator's request, the next single-parameter candidate is the midpoint
`0.50 N/update`. Frozen replay of `experimental_auto_20260814_234007.jsonl`
predicts forward base-lag P50/P95/max of `0.000/1.353/2.415 N`, with `48.9%`
of eligible updates at the limit. This is only an open water-test candidate;
it has not yet demonstrated stationary tracking or constant-speed following.
Right/down, all MPC weights, force-rate limits, PID, and vision settings remain
unchanged.

The water run `experimental_auto_20260814_235737.jsonl` rejected the `0.50`
candidate based on both operator observation and the stationary ending. Over
the last ten seconds, forward error RMS/P95 remained `9.67/15.59 cm`, while
`10.5%` of forward commands were saturated. Across the full trace, `56.8%`
of eligible base updates reached `0.50 N` and forward base lag P50/P95/max was
`0.130/1.764/3.286 N`. The operator judged tracking worse than `0.40`, so the
active forward limit is restored to `0.40 N/update`. No other parameter is
changed.

The next water run must again begin with an explicitly announced stationary
10 s interval before one slow constant-speed pass and stop. Compare base lag,
static hold, cruise error, and stopping overshoot against this trace.

## 2026-08-14: moderate forward force-change smoothing after Q=800 run

The translation-only run `experimental_auto_20260814_225507.jsonl` tested
`Q_forward=800` with `R_force,forward=0.25`. Synchronized overhead data showed
that during positive motion at approximately `0.05--0.056 m/s`, the vehicle
reached approximately `0.058--0.061 m/s` and forward MAE was approximately
`2.27--3.36 cm`. The preceding stationary interval had forward MAE/P95 of
approximately `2.19/4.11 cm`. The higher position weight therefore materially
reduced constant-speed lag without immediately losing stationary hold.

The reverse interval still produced a five-second forward MAE/P95 of
approximately `10.88/17.75 cm`, with the forward channel at its `20%` cap in
approximately `19%` of that window. The operator also observed excessive
speed variation. Frozen QP replay of the complete trace compared forward
delta-force weights `4/6/8`: mean absolute first-step force change was
`0.331/0.287/0.257 N`, and use of the `0.8 N` per-step limit was
`13.6/9.1/7.0%`. The `8` candidate also reduced the P95 second-step force
change from `0.800 N` to `0.656 N`, creating a larger risk of delayed reversal
braking.

The water run `experimental_auto_20260814_230123.jsonl` rejected the `6.0`
candidate. Although its first settled stationary window briefly reached about
`1.5--1.7 cm` forward MAE, after target motion the nominally stationary target
produced repeated low-frequency chase cycles. Five-second windows reached
approximately `20--54 cm` forward P95 while forward model-1 weight rose to
about `0.62--0.82`; the latched forward base retained approximately
`-2.5--+1.4 N` from the motion phase. Increasing the change penalty delayed
unloading/reversal of this stale operating force and did not solve speed
variation. The forward delta-force weight is therefore restored to `4.0`.
Do not repeat `6.0` or increase it to `8.0` for this symptom. Position,
velocity, and force weights remain `800/150/0.25`, horizon remains `10`, and
fusion, limits, PID, and vision remain unchanged. The next controller change
must address model-1 operating-force credibility during stopping, preferably
by suppressing model 1 when the achieved-force baseline is varying or
reversing while preserving it under a stable cruise force.

This run also ended with `armed command confirmation stale`, the second such
event during a visual-result gap. Both post-stop disarmed link checks showed
fresh telemetry, accepted zero commands, state `IDLE`, armed false, and zero
motor outputs. Do not weaken the `0.25 s` confirmation guard; reduce host-side
load/output backpressure and retain the safety stop before another armed run.

## 2026-08-14: raise forward position weight after R=0.25 water test

The translation-only run `experimental_auto_20260814_224318.jsonl` tested the
preceding single change `R_force,forward=0.25`. The synchronized overhead bag
`top_camera_20260814_224228` showed that slow positive motion had forward MAE
`1.79 cm`, and a stationary interval had MAE/P95 `1.08/2.32 cm`; static hold
therefore remained good. At approximately `0.11--0.12 m/s` target speed,
however, two fast constant-direction intervals still had signed forward error
of approximately `+9.14 cm` and `-10.84 cm`. Lowering only the force penalty
was therefore insufficient to meet the requested `5 cm` cruise-error target.

Across the full run, the forward channel reached its `20%` cap in only `2.5%`
of control updates, the maximum final motor magnitude was `0.335`, and no
motor approached the `0.50` final cap. A frozen QP replay of the same logged
states with the already-active `R_force,forward=0.25` raised first-step
forward `0.8 N` delta-force-bound use from `10.6%` at `Q_forward=500` to
`14.1%` at `Q_forward=800`; force-bound use rose only from `1.2%` to `2.1%`.
Only `position_weights[forward]` is therefore changed from `500` to `800`.
Forward velocity/force/delta-force weights remain `150/0.25/4.0`, horizon
remains `10`, and fusion, limits, PID, and vision remain unchanged. The next
run must again check stationary hold first, then slow and fast straight
forward/reverse passes, with special attention to reversal overshoot.

The same run ended with the independent safety fault `armed command
confirmation stale` after a visual-quality gap. The subsequent disarmed link
test confirmed state `IDLE`, armed false, zero requested/received channels,
zero applied motor outputs, no failsafe, and no command rejection. This was
not a QP failure and does not justify weakening the `0.25 s` command
confirmation safety limit.

## 2026-08-14: lower forward equilibrium-relative force penalty

After rejecting the frozen high model-1 weight experiment, the adaptive
fusion threshold was restored to `1e-7 m^2`. Steady-state QP solves were then
performed for forward model-1 weights `0.20/0.35/0.50` and latched bases
`1.0/1.5/2.0 N`. With the current `Q_forward=500` and `R_force,forward=0.50`,
the representative `a1=0.35`, `tau_base=1.5 N` equilibrium required
approximately `7.92 cm` position error before the MPC requested the full base.
This matches the measured fast constant-speed lag.

Raising `Q_forward` to `800` reduced that calculated offset to `5.31 cm`;
lowering only `R_force,forward` to `0.25` produced a comparable `5.48 cm`.
Frozen replay over `experimental_auto_20260814_215909.jsonl` showed the Q=800
candidate increasing first-step `0.8 N` delta-force-bound use from `5.8%` to
`9.3%`, while the R=0.25 candidate increased it only to `6.2%`. Force/channel
cap fractions were similar (`5.3%` versus `5.2%`). Therefore only the forward
force weight changes from `0.50` to `0.25`. Forward position/velocity weights
remain `500/150`, delta-force weight remains `4.0`, horizon remains `10`, and
fusion, limits, PID, and vision remain unchanged. The next water run must
first verify a stationary target, then compare slow and fast constant-speed
errors and reversal overshoot.

## 2026-08-14: forward fusion identifiability threshold after variable-speed passes

The translation-only run `experimental_auto_20260814_215909.jsonl`, aligned
with overhead bag `top_camera_20260814_215826`, contained repeated slow and
fast forward/reverse passes. Long passes near `0.05 m/s` had mean absolute
forward error of approximately `3.6--4.2 cm`, while passes near
`0.10--0.13 m/s` rose to approximately `7.8--9.8 cm`. The vehicle and target
mean speeds were already close in those fast passes, but the forward model-1
weight remained only about `0.34--0.42`.

At fast steady speed, the latched forward base was commonly `1.6--1.8 N`, the
fused equilibrium reference was only `1.1--1.2 N`, and requested force was
only `0.01--0.06 N` beyond the latched base. The controller therefore matched
speed without producing enough excess force to close the accumulated
`7--9 cm` error promptly. Forward channel saturation occurred in only about
`6--17%` of the selected fast-pass samples and final motor throttle peaked at
`0.409`, so the final motor limit was not the primary cause.

The logged forward candidate-separation score corresponded to only about
`0.94 mm RMS` on average and `3.21 mm` at P95, versus the configured forward
position measurement standard deviation of `30 mm`. The previous forward
indistinguishability threshold, `1e-7 m^2` (`0.316 mm RMS`), allowed the fusion
weight to react to candidate differences far below measurement noise. Frozen
replays with fusion horizons 5 and 8 did not make model 1 consistently more
identifiable. Only the forward `indistinguishable_score_threshold` is changed
from `1e-7` to `1e-4 m^2` (`10 mm RMS`) for one explicit A/B test. This
deliberately preserved the existing `0.80` forward model-1 prior when the
candidates lacked meaningful separation; right/down thresholds, MPC costs,
horizon, limits, PID, and vision remained unchanged.

The A/B run `experimental_auto_20260814_221332.jsonl` rejected this change.
With the target held approximately stationary, the forward model-1 weight
remained exactly `0.80`; forward error swung over approximately
`-28.6--+32.0 cm`, mean absolute error rose to `12.9 cm`, and the forward
channel hit its `20%` limit in `81/344` updates. The operator directly observed
that the controller could no longer track even the stationary target. Holding
the previous achieved force as a high-weight equilibrium therefore retains
too much force during stopping and creates a chase/overshoot/reversal cycle.
The forward threshold was immediately restored to `1e-7 m^2`. Do not repeat
the `1e-4` threshold experiment; the next iteration must reduce the model-1
operating-force memory during stopping rather than freezing model 1 at a high
weight.

## 2026-08-14: forward zero-crossing damping after equilibrium-relative force cost

The translation-only run `experimental_auto_20260814_214726.jsonl`, aligned
with overhead bag `top_camera_20260814_214614`, contained three approximately
constant-speed forward/reverse intervals. The overhead target moved at
`0.063--0.074 m/s` while the vehicle moved at `0.057--0.066 m/s`. Signed
steady forward error was approximately `4.3 cm`, `5.4 cm`, and `0.4 cm`, a
large improvement over the earlier roughly `20 cm` lag after centering the
force cost on the fused equilibrium. The operator nevertheless observed
excess zero-crossing overshoot and requested at most `5 cm` error during
constant-speed motion.

Only the forward velocity weight was changed from `100` to `150`. Forward
position weight remains `500`, forward force/delta-force weights remain
`0.5/4.0`, and horizon remains `10`. This adds damping to relative forward
velocity without moving the zero-error equilibrium or simultaneously changing
the force slew penalty. No firmware or vision change is required.

## 2026-08-06: faster fish tracking

The Unity integration uses symmetric positive and negative limits. No
direction-dependent compensation was added.

| Parameter | Previous value | Current value |
| --- | --- | --- |
| `horizon` | `8` | `10` |
| `reference_position` | `(0.80, 0.0, 0.0)` | `(0.80, 0.0, 0.0)` |
| `position_weights` | `(120.0, 420.0, 220.0)` | `(260.0, 700.0, 380.0)` |
| `velocity_weights` | `(8.0, 12.0, 12.0)` | `(10.0, 16.0, 16.0)` |
| `force_weights` | `(0.04, 0.03, 0.06)` | `(0.025, 0.018, 0.04)` |
| `delta_force_weights` | `(0.8, 0.45, 1.0)` | `(0.45, 0.25, 0.65)` |
| `force_min` | `(-28.0, -24.0, -15.0)` | `(-34.0, -32.0, -20.0)` |
| `force_max` | `(28.0, 24.0, 15.0)` | `(34.0, 32.0, 20.0)` |
| `delta_force_min` | `(-4.0, -3.2, -2.0)` | `(-6.0, -4.8, -3.0)` |
| `delta_force_max` | `(4.0, 3.2, 2.0)` | `(6.0, 4.8, 3.0)` |
| `ForceCommandAdapter.positive_force_at_limit` | `(28.0, 24.0, 15.0)` | `(34.0, 32.0, 20.0)` |
| `slack_quadratic_weight` | default `2.0e4` | `5.0e4` |
| `slack_linear_weight` | default `50.0` | `100.0` |

The current set produced mean absolute forward-distance errors of `0.118 m`
while the fish moved in +X and `0.120 m` while it moved in -X during a 45-second
Unity test. All 901 OSQP samples solved without fallback.

## 2026-08-06: physical force envelope from 7 N thrusters

The `FinsROV.prefab` has four horizontal thrusters mounted at 45 degrees and
four vertical thrusters. Every thruster is configured for symmetric `+7/-7 N`.
Consequently, the maximum pure body force is:

- forward: `4 * 7 * cos(45 deg) = 19.799 N`
- right: `4 * 7 * cos(45 deg) = 19.799 N`
- down: `4 * 7 = 28.0 N`

This set replaces only the force envelope before collecting new tuning data.
The tracking weights remain those from the preceding tuning pass.

| Parameter | Previous value | Current value |
| --- | --- | --- |
| `force_min` | `(-34.0, -32.0, -20.0)` | `(-19.799, -19.799, -28.0)` |
| `force_max` | `(34.0, 32.0, 20.0)` | `(19.799, 19.799, 28.0)` |
| `delta_force_min` | `(-6.0, -4.8, -3.0)` | `(-4.0, -4.0, -5.6)` |
| `delta_force_max` | `(6.0, 4.8, 3.0)` | `(4.0, 4.0, 5.6)` |
| `ForceCommandAdapter.positive_force_at_limit` | `(34.0, 32.0, 20.0)` | `(19.799, 19.799, 28.0)` |

Baseline data with the physical envelope and `position_weights[forward] = 260`:

- duration: `68.7 s` / `1374` control samples
- forward MAE / maximum: `0.133 / 0.200 m`
- reverse MAE / maximum: `0.128 / 0.198 m`
- forward / reverse force saturation: `20.4% / 16.8%`
- QP solve success: `100%`

## 2026-08-06: forward tracking weight iteration 1

Only the forward position weight changes in this iteration. All force and
force-rate limits remain at the 7 N thruster-derived values above.

| Parameter | Previous value | Current value |
| --- | --- | --- |
| `position_weights` | `(260.0, 700.0, 380.0)` | `(520.0, 700.0, 380.0)` |

Iteration 1 result (`69.0 s`, `1381` samples):

- forward MAE / maximum: `0.098 / 0.170 m`
- reverse MAE / maximum: `0.091 / 0.174 m`
- overall forward-error MAE improvement: `28.3%`
- force saturation: `21.7%` (baseline `18.5%`)
- QP solve success: `100%`

## 2026-08-06: forward tracking weight iteration 2

| Parameter | Previous value | Current value |
| --- | --- | --- |
| `position_weights` | `(520.0, 700.0, 380.0)` | `(800.0, 700.0, 380.0)` |

Iteration 2 result (`69.0 s`, `1381` samples):

- forward MAE / maximum: `0.086 / 0.162 m`
- reverse MAE / maximum: `0.082 / 0.197 m`
- force saturation: `29.2%`
- QP solve success: `100%`

Although the overall MAE decreased by another `10.3%`, saturation increased by
`7.5` percentage points and the reverse maximum error became `13.3%` worse.
Therefore iteration 2 was rejected and the setting selected for the `0.18 m/s`
test was restored to `position_weights = (520.0, 700.0, 380.0)`.

## Selected current configuration

| Parameter | Selected value |
| --- | --- |
| `horizon` | `10` |
| `reference_position` | `(0.80, 0.0, 0.0)` |
| `position_weights` | `(1600.0, 700.0, 15000.0)` |
| `velocity_weights` | `(18.0, 16.0, 16.0)` |
| `force_weights` | `(0.012, 0.018, 0.04)` |
| `delta_force_weights` | `(0.08, 0.25, 0.65)` |
| `force_min/max` | `+/- (19.799, 19.799, 28.0) N` |
| `delta_force_min/max` | `+/- (4.0, 4.0, 5.6) N/sample` |
| `slack_quadratic_weight` | `5.0e4` |
| `slack_linear_weight` | `100.0` |

## 2026-08-06: calibration audit and fish-speed reduction

The `CALIBRATION_CHECKLIST.md` audit separates parameters verified for the
Unity ground-truth integration from simulation assumptions and real-hardware
calibration work. In particular, `M_t`, `D_L`, camera extrinsics, measurement
noise, and total vision latency remain uncalibrated placeholders.

The straight-line fish speed was reduced from `0.18 m/s` to `0.12 m/s`. At
`0.18 m/s`, the selected `position_weights[forward] = 520` test used roughly
`17.2 N` mean absolute forward force, about `87%` of the `19.799 N` axis limit,
and saturated for `21.7%` of samples. The lower speed is intended to restore
control headroom before the next weight iteration.

The OSQP time limit was also reduced from `0.08 s` to `0.035 s`, so it is now
strictly shorter than the `0.05 s` control period.

## 2026-08-06: tuning at 0.12 m/s fish speed

Both runs contain `1372` samples over `68.6 s`, cover positive and negative X
motion, and retain the same symmetric 7 N thruster-derived force limits. Only
the forward position weight differs.

| Metric | `qx=520` baseline | `qx=650` candidate |
| --- | --- | --- |
| Positive-X error MAE / maximum | `0.0680 / 0.1037 m` | `0.0620 / 0.0976 m` |
| Negative-X error MAE / maximum | `0.0691 / 0.1149 m` | `0.0632 / 0.1107 m` |
| Overall error MAE | `0.0676 m` | `0.0619 m` |
| Mean absolute forward force | `11.28 N` | `11.50 N` |
| Peak absolute forward force | `17.47 N` | `19.799 N` |
| Forward-force saturation | `0.00%` | `0.15%` |
| QP solve success | `100%` | `100%` |

The `qx=650` candidate improves overall forward error MAE by `8.3%` for only
`0.22 N` additional mean force. Its two approximate saturation samples are
accepted, so the selected position weights are now `(650, 700, 380)`. The
lateral weight is not changed because the straight-X fish path provides no
lateral excitation for a defensible comparison.

## 2026-08-06: superseded target-motion feedforward experiment

UnderwaterVision uses a `26.07276 kg` Rigidbody and DWP2 `WaterObject`
hydrodynamics. Three-axis `+/-6 N` and `+/-12 N` step responses were fitted
directly against the complete velocity response. In MPC forward/right/down
order (Unity `X/Z/-Y`), the selected simulation model is:

- `M_t = diag(26.07276, 26.79684, 26.07276) kg`
- `D_L = diag(93.88006, 143.69195, 280.86849) N/(m/s)`
- fixed initial `tau_base = (0, 0, 1.89299) N`

The axis velocity-fit RMSE values are `0.00276`, `0.00320`, and `0.00211 m/s`,
with `R²` values `0.9986`, `0.9956`, and `0.9928`. These are simulation-only
equivalent parameters, not FineSUB real-hardware calibration results. The raw
fit is stored in `dynamics_20260806_212129.json`.

The tracker estimator was corrected to propagate absolute force as
`B @ (tau - tau_base)`. Target velocity now uses the Unity message timestamp,
and the MPC prediction model includes the target-motion equilibrium force
`tau_base + D_L @ target_velocity + M_t @ target_acceleration`. This is not an
output-side compensation term. The fish follows a straight `3 m` ping-pong
path at up to `0.12 m/s`, with approximately `0.08 m/s²` smooth turnarounds.

The final tuning retained every physical force and force-rate limit. Only the
forward delta-force cost changed from `0.45` to `0.15`; the down-position
weight is `15000` to reject the identified upward environmental bias.

Final result (`73.55 s`, `1471` samples):

- 3D error mean / maximum: `0.00739 / 0.02492 m`
- samples below `0.03 m`: `1471 / 1471` (`100%`)
- turnaround error mean / maximum: `0.01744 / 0.02492 m`
- peak forward force: `14.90 N` of the unchanged `19.799 N` limit
- OSQP success: `100%`

Artifacts are `identified_mpc_tracking_final_decoupled_20260806.csv` and
`identified_mpc_tracking_final_decoupled_20260806.png` in the ROS workspace
`data/mpc_tracking` directory.

This result is retained for comparison only. It used separate Unity world
poses for the fish and ROV, derived absolute target velocity/acceleration, and
smooth endpoint turnarounds. Those inputs exceed the real controller's stated
sensor boundary, so this run is not a valid acceptance result for the current
integration.

## 2026-08-06: observable-only relative-position interface

The active interface now exposes only an ideal body-FRD target-relative
position vector and its message timestamp. The translation MPC no longer
subscribes to ROV world pose, fish world pose, DVL, target absolute velocity,
or target absolute acceleration. Its Kalman filter estimates relative velocity
from relative-position history. The prior saturated force command remains
available internally as an approximation of achieved force.

`M_t` and `D_L` remain fixed at the identified simulation constants for the
entire run. `tau_base` now starts at `(0,0,0) N` and the observable adaptation
is explicitly enabled. The fish endpoint behavior is again an instantaneous
reversal. Acceptance is based on constant-speed steady-state error below
`0.03 m`, not the unavoidable reversal transient.

## 2026-08-06: observable baseline deadlock fix and cost sweep

The original matched-state gate used three-axis norms and required total
position error below `0.06 m` and relative speed below `0.04 m/s`. Once a fish
reversal produced about `0.14 m` error, `tau_base` could no longer update from
the old `-11.4 N` equilibrium to the required positive equilibrium. Baseline
learning therefore deadlocked even though the optimizer continued solving.

Baseline adaptation now evaluates each axis independently, uses rate `0.03`,
position tolerance `0.20 m`, and relative-velocity tolerance `0.08 m/s`, and
clips the learned baseline to the unchanged physical force limits. It still
uses only relative-position-derived state and the preceding applied command;
no fish/ROV world pose or absolute target velocity is exposed. `M_t` and `D_L`
remain fixed.

Each cost candidate contains `1301` samples over `65.0 s`, covers both fish
directions and instantaneous reversals, and has `100%` QP solve success.

| Metric | Deadlocked baseline | Candidate 1 | Candidate 2 |
| --- | ---: | ---: | ---: |
| Forward position / velocity weight | `650 / 10` | `1600 / 18` | `2400 / 22` |
| Forward force / delta-force weight | `0.025 / 0.15` | `0.012 / 0.08` | `0.008 / 0.05` |
| Overall forward-error MAE | `0.07859 m` | `0.01246 m` | `0.01480 m` |
| Overall forward-error 95th percentile | `0.19864 m` | `0.09203 m` | `0.04527 m` |
| Steady forward-error MAE | deadlocked in one direction | `0.00448 m` | `0.01234 m` |
| Steady forward-error 95th percentile | deadlocked in one direction | `0.01185 m` | `0.01980 m` |
| Peak reversal error | `0.24510 m` | `0.14901 m` | `0.11803 m` |
| Forward saturation samples | `65` | `83` | `99` |

Candidate 2 reduces the instantaneous reversal peak but worsens steady error
and produces visibly larger force oscillation. Candidate 1 is selected because
the acceptance target concerns constant-speed tracking and its steady 95th
percentile remains below `0.012 m`, with both learned directional equilibria
converging near `+/-11.3 N`.

## 2026-08-06: baseline update strategy and rate comparison

The bridge now supports two explicit baseline modes. `gated_ema` is the normal
mode. `previous_force` is retained only to reproduce the requested comparison;
it assigns the preceding command directly to `tau_base` every sample.

The direct `previous_force` experiment was stopped after 201 samples because
it was clearly unstable. Its mean 3-D error was `0.14469 m`, compared with
`0.01380 m` for gated EMA. At least one axis saturated in 180/201 samples, and
all three `tau_base` axes traversed their complete physical limits. It must not
be used as the default because feedback corrections are not a slowly varying
motion equilibrium.

Four gated EMA rates were then compared over 1301 samples / 65 seconds each:

| Baseline rate | Overall X MAE | Steady X MAE | Steady X p95 | Saturation samples |
| --- | ---: | ---: | ---: | ---: |
| Fixed `0.03` | `0.01246 m` | `0.00352 m` | `0.00803 m` | `83` |
| Fixed `0.015` | `0.01293 m` | `0.00243 m` | `0.01000 m` | `61` |
| Fixed `0.01` | `0.01980 m` | `0.00157 m` | `0.00470 m` | `118` |
| Adaptive `0.03 -> 0.01` | `0.01389 m` | `0.00128 m` | `0.00367 m` | `95` |

The adaptive rate is selected because acceptance prioritizes constant-speed
error rather than the unavoidable instantaneous-reversal peak. Each axis uses
rate `0.03` while its absolute position error exceeds `0.03 m` or estimated
relative speed exceeds `0.02 m/s`; it switches to `0.01` inside that steady
region. Baseline eligibility remains independently gated per axis at `0.20 m`
and `0.08 m/s`. The unchanged force envelope is `+/-19.799 N` forward/right
and `+/-28 N` down.

## 2026-08-07: independently tuned direct and indirect baseline strategies

The two baseline update strategies were tuned independently in Unity rather
than evaluating `previous_force` with weights selected for `gated_ema`. Every
screening trial started from a fresh UnderwaterVision Play session. The fish
used the same 3 m straight-X PingPong trajectory at 0.12 m/s with instantaneous
reversals every 25 seconds.

Five 40-second direct `previous_force` candidates were tested. Increasing
velocity and force-rate penalties reduced the original immediate instability,
but no candidate produced two continuous seconds with estimated relative speed
inside +/-0.02 m/s. The selected direct weights are:

| Weight | Forward / Right / Down |
| --- | --- |
| Position | `8000 / 1000 / 20000` |
| Velocity | `3000 / 1000 / 20000` |
| Force | `0.2 / 0.5 / 2.0` |
| Delta force | `4 / 8 / 40` |

Three 40-second indirect `gated_ema` candidates were tested around the previous
selection. More aggressive position tracking and additional velocity damping
both worsened either steady error or reversal recovery. The retained indirect
weights remain:

| Weight | Forward / Right / Down |
| --- | --- |
| Position | `1600 / 700 / 15000` |
| Velocity | `18 / 16 / 16` |
| Force | `0.012 / 0.018 / 0.04` |
| Delta force | `0.08 / 0.25 / 0.65` |

The selected candidates were then rerun for 65 seconds / about 1301 samples,
covering two reversals. A fixed constant-speed window excludes eight seconds
after startup and each reversal; it does not select samples based on error.

| Metric | Direct `previous_force` | Indirect `gated_ema` |
| --- | ---: | ---: |
| Full-run X MAE | `0.06987 m` | `0.01699 m` |
| Constant-speed X MAE / p95 | `0.05831 / 0.12454 m` | `0.00203 / 0.00644 m` |
| Constant-speed 3-D MAE / p95 | `0.14034 / 0.25513 m` | `0.00269 / 0.00767 m` |
| Error-norm samples <= 0.03 m | `0.92%` | `86.10%` |
| At least one saturated axis | `46.20%` | `9.98%` |
| Strict matched-speed samples | `0` | `915` |
| QP solve success | `100%` | `100%` |

Indirect baseline adaptation reduces constant-speed X MAE by `96.5%` and 3-D
MAE by `98.1%`. The default remains `gated_ema`; the tuned direct profile is
retained for reproducible comparison, not selected for normal operation.
## 2026-08-07: restore pure model-2 branch and enforce joint thruster envelope

The maintained path remains `MPC_dual_model`. The fused prediction now exactly
uses the two candidate structures:

```text
model 1: x+ = A_d x + B_d (tau - tau_base)
model 2: x+ = A_d x + B_d tau
```

The previous change that subtracted `tau_base` in model 2 was removed. The
baseline is now used only by model 1, observable baseline learning, and safe
fallback; it is not a privileged fish-motion input and never enters model-2
prediction. The bridge publishes both model weights, fusion sample count, and
normalized thruster utilization so that constant-speed behavior can be checked
directly. The maintained fusion also updates the model parameter with rate
`0.20`, so close residuals cannot make the controller switch candidates every
sample. At that time the screening configuration used `minimum_weight=0.05`;
the maintained configuration is now documented in the current-baseline entry
below and uses `minimum_weight=0.01`, so a stable axis can reach `0.99`.

The QP now also constrains the eight FineSUB translation commands. The four
horizontal 45-degree thrusters share the joint envelope
`|F_forward|/19.799 + |F_right|/19.799 <= 1`; the four vertical thrusters retain
`|F_down| <= 28 N`. This replaces the earlier assumption that all axis maxima
could be reached simultaneously. This entry recorded an earlier `0.08 m/s`
screening run; it is not the current Unity scene value. Automatic trials then
opened the live error/weight plot by default and saved it to
`ros2_ws/mpc_tracking_live.png`.

## 2026-08-07: current dual-model curve baseline and diagnostic observability

The effective Unity baseline is the `UnderwaterVision` scene configuration:
horizontal `StraightXWithZSin`, `xTravelDistance=0.75 m`,
`movementSpeed=0.015 m/s`, `zSinAmplitude=0.10 m`, `sinCycles=1`, constant
depth, and instantaneous PingPong reversal. The 65 s `gated_ema` trial is the
current comparison baseline: 1301 samples, 20 Hz, 3-D error mean `0.00857 m`,
P95 `0.02072 m`, constant-speed 3-D error mean `0.00579 m`, P95 `0.01520 m`,
98.23% of samples within `0.03 m`, QP success `100%`, and zero saturation.

The dual-model equations remain exactly:

```text
model 1: x+ = A_d x + B_d (tau - tau_base)
model 2: x+ = A_d x + B_d tau
```

`tau_base` is not target feedforward and is not visible to Unity as an extra
measurement. It is learned only from the observable relative position, its
filtered relative velocity, and the previous applied command approximation;
it enters model 1 and safe fallback only. A per-axis candidate-difference
floor of `1e-7 m2` now holds the previous fusion weight when the two residual
models are indistinguishable at the current measurement scale. Neutral
evidence recovers toward model 1, preventing model-2 noise switching during
constant-speed/constant-depth tracking while preserving model-2 selection for
a clearly better rapid-change prediction.

The diagnostics message keeps the original 33 fields and appends 9 fields:
right/down relative velocity, 3-D speed, three relative accelerations, and the
three candidate-difference scores. Each automatic trial now writes its own
live PNG next to its CSV, while manual launches continue to update
`ros2_ws/mpc_tracking_live.png`.

## 2026-08-07: joint-envelope horizontal curve tuning

The final pressure trajectory is a horizontal XZ sine PingPong path with
`xTravelDistance=0.75 m`, `movementSpeed=0.030 m/s`, `zSinAmplitude=0.08 m`,
and two sine cycles per one-way traversal. Endpoint reversal remains
instantaneous. The selected FRD costs are:

| Weight | Forward / Right / Down |
| --- | --- |
| Position | `1600 / 3000 / 15000` |
| Velocity | `18 / 30 / 16` |
| Effective force | `0.025 / 0.012 / 0.06` |
| Delta force | `0.12 / 0.06 / 0.8` |

The 65-second trial recorded 1302 samples at 20 Hz. Full-run 3-D error mean
was `0.0193 m`; strict matched-relative-speed error mean/P95 was
`0.0096/0.0268 m`. The horizontal joint envelope reached exactly `1.0` for
four startup samples at `0.20--0.35 s` and remained below the limit afterward,
including both instantaneous endpoint reversals. No command exceeded the
joint thruster envelope.

All three model-1 weight medians were approximately `0.99`. In the highest
15 percent estimated-relative-acceleration samples, the mean forward/right
model-1 weights fell to `0.935/0.463`; model 2 had lower forward/right MSE in
that segment. Thus constant-motion behavior remains model-1 dominant while
the rapid-change samples retain a measurable model-2 contribution.

## 2026-08-08: maximum confirmed S-trajectory speed

The Unity project now lives at
`/home/fins/Zhouyuheng_workspace/marus-example`. The selected scene uses
`xTravelDistance=0.75 m`, `movementSpeed=0.11 m/s`, `zSinAmplitude=0.04 m`,
one sine cycle per one-way traversal, constant depth, and instantaneous
PingPong reversal. The selected FRD costs are:

| Weight | Forward / Right / Down |
| --- | --- |
| Position | `10000 / 5000 / 25000` |
| Velocity | `2 / 20 / 12` |
| Effective force | `0.003 / 0.008 / 0.04` |
| Delta force | `0.01 / 0.04 / 0.5` |

The final 65-second trial recorded 1302 samples at 20 Hz. Continuous settled
tracking had 3-D error mean/P95 `0.0178/0.0238 m`. After the first five
seconds, the 626 samples with estimated relative speed at or below `0.04 m/s`
had mean/P95 `0.0174/0.0298 m`. Instantaneous reversals are intentionally not
required to remain below `0.03 m`.

The horizontal joint force envelope was active for `27.6%` of samples but was
never exceeded. A `0.115 m/s` boundary trial produced no continuous settled
samples, and `0.12 m/s` saturated the horizontal envelope for `60.5%` of the
trial without settling. Therefore `0.11 m/s` is the highest confirmed speed
for this trajectory and the current physical force limits.

## 2026-08-08: V4 Pro1 asymmetric limits and racetrack tuning

The current QP replaces the historical symmetric `+/-7 N` approximation with
the V4 Pro1 canonical per-thruster positive/negative `force_n` limits. The
resulting pure-axis bounds are `(-16.2968,-16.2968,-23.0472) N` to
`(16.2968,16.2968,29.5236) N`; all eight asymmetric row constraints remain in
the QP, so diagonal requests are further coupled.

The Unity fish now follows a continuous horizontal racetrack with `0.75 m`
straight sections, `0.20 m` semicircle radius, constant depth, and a `2.0 s`
start delay aligned with relative-target publication. The selected speed is
`0.0975 m/s`, and the selected FRD costs are position
`(10000,14000,25000)`, velocity `(2,20,12)`, effective force
`(0.003,0.002,0.04)`, and delta force `(0.01,0.01,0.5)`.

The final 65-second run recorded 1302 samples. Matched-relative-speed error
mean/P95 was `0.01237/0.02665 m`; low-acceleration P95 was `0.01655 m`,
settled P95 was `0.01032 m`, solver success was `100%`, and the exact
per-thruster horizontal envelope was active for `23.1%` of samples. A
`0.10 m/s` run with the same final weights reached matched P95 `0.03205 m`,
so `0.0975 m/s` is the highest confirmed speed under these limits.

## 2026-08-13--14: real-device 0.6 m reference and structural tuning audit

The real-device camera reference is now body FRD
`[0.857634,-0.055545,-0.120815] m`, corresponding to `0.6 m` along the
onboard camera optical axis. Six weight/rate trials were collected. Lowering
the sway position weight from `700` to `350` reduced command magnitude but did
not remove opposite-side overshoot. Lowering only the sway delta-force cost
from `2.0` to `0.5` improved the two observed sway reversals from roughly
`44--45%` overshoot to about `22--32%`. Lowering the surge delta-force cost to
`0.5` produced about `84%` surge overshoot and was rejected. The current surge
delta-force cost `4.0` has not yet received a clean excitation test.

The follow-up audit found that weight-only tuning is premature:

- `N=5` spans only `0.5 s`, versus model `M/D` time constants of
  `1.755--2.588 s` and measured RPM spin-down tails up to about `0.8 s`;
- `5.0--7.5%` of accepted update intervals exceeded `0.15 s`, while the
  estimator always advances one fixed `0.10 s` step;
- the replayed learned horizontal baseline reached ranges of approximately
  `1.5--2.4 N` and can persist when a hand-moved target stops;
- the final `0.50` motor cap was never reached, but mixed-channel distortion
  above `0.005` occurred in `23--62%` of updates;
- the `0.01` software motor deadband corresponds to a yaw P-only angular
  threshold of about `1.53 deg` with the selected lower PID;
- RPM-curve and command-linear achieved-force estimates differed by roughly
  `0.44--1.21 N` at P95, so both must be logged before selecting the feedback
  source.

In the deterministic current-model benchmark, diagonal-step overshoot for
`N=5/10/15/20/25` was approximately `8.8/11.9/6.8/3.8/2.8%`, while P95 solve
time was `2.1/2.9/4.2/5.3/7.3 ms`. Longer horizons also requested more force,
and a noisy delayed/dropout simulation showed that an aggressive `N=20`
candidate is unsafe to select before fixing the time base. The first post-fix
real-device candidate is therefore a damped `N=15` bundle, not an immediate
configuration change. Full parameter inventory, symptom routing, candidate
values and test order are in `REAL_DEVICE_TUNING_GUIDE.md`; reproducible audit
output is `calibration_logs/offline_tuning_audit_20260814.analysis.json`.

## 2026-08-14: corrected fixed-per-solve base implementation

The earlier slow-horizontal-disturbance interpretation of `tau_base` was
wrong, but the first audit correction then made a second mistake: it claimed
that model 1 must replace its base with the preceding planned force at every
future horizon step. The confirmed PDF semantics are instead:

```text
tau_base,k = tau_achieved,k-1
v1[j+1] = F v1[j] + G (tau[j] - tau_base,k)
```

`tau_base,k` is fixed for the complete solve that starts at `k`; the next solve
latches a new achieved force. Consecutive planned forces still roll for the
separate rate cost and constraint, `Delta tau[j]=tau[j]-tau[j-1]`, but they do
not replace model 1's fixed origin.

The active dual implementation now follows that definition. The old
`gated_ema` path has been removed from active tracking, fixed
`tau_h=h(0)=[0,0,0.807290] N` is stored separately, and model 2 uses
`v2+=Fv+G(tau-tau_h)`. Kalman mean prediction, historical model scoring and
the QP use the same two candidates. The QP cost is absolute total force plus
consecutive force change, PDF position propagation uses the end-of-step
relative velocity, indistinguishable fusion evidence holds the previous
weight, and solver/vision-gap fallback returns toward `tau_h`. The earlier
recommendation to disable horizontal baseline remains withdrawn.

Remaining structural gaps at that checkpoint were fixed versus actual visual
`dt`, actuator delay/inertia, command-derived achieved force, active-yaw frame
rotation, camera-coordinate FOV constraints, and the experimental identity
thruster envelope. See `MODEL_PDF_IMPLEMENTATION_AUDIT.md`.

## 2026-08-14: RPM achieved force, yaw-frame prediction, and camera constraints

The active formal/experimental AUTO adapter now reconstructs achieved force
and yaw moment from signed DSHOT RPM using the accepted same-vehicle
positive/negative quadratic curves and CAD force geometry. Horizontal
M1/M2/M6/M7 and vertical M3/M4/M5/M8 valid bits are gated independently; an
incomplete group falls back only that group's final applied motor set-points.
Per-thruster curve limits clamp invalid RPM magnitude excursions, and the
trace records the selected source and each reconstructed thruster force.

Between visual updates, the estimator and every pending fusion prediction now
rotate old body coordinates by the measured telemetry yaw increment. Inside
the QP horizon, each step rotates coordinates using the current body-FRD yaw
rate under a constant-rate assumption. The raw H30 telemetry rate sign is
converted to the same FRD convention already used by `V5_SUB`.

FOV and range constraints now subtract the body-frame camera origin and apply
the active camera rotation before constraining
`[camera-forward,camera-right,camera-down]`. The experimental `I3` thruster
constraint is intentionally documented but not changed in this revision: it
duplicates the three translation-axis boxes and still omits actuator sharing
with local yaw and roll/pitch PID.

## 2026-08-14: RPM achieved-force control rollback after water test

The first post-change water run accepted only two visual updates. A preceding
`+0.0223` forward / `+0.0327` right channel was reconstructed from
instantaneous RPM as `+1.138 N` forward / `+3.097 N` right, and the next MPC
request increased to `+0.0563/+0.0954`. The target then crossed the configured
forward-range gate and the later QP became primal infeasible while using the
RPM force as both fixed model-1 base and hard slew reference.

RPM is therefore diagnostic-only again. Active `tau_achieved` uses the final
mixed motor-throttle echo, while the signed per-thruster and combined RPM
force estimates remain in the trace for offline comparison. Yaw-frame
rotation, camera-coordinate visibility constraints, fixed-per-solve
`tau_base`, and the independent Fossen restoring force are unchanged.

## 2026-08-14: throttle-feedback retry and experimental vision range

The retry `experimental_auto_20260814_172803.jsonl` ran for about `292 s` and
completed `1928/1928` QP updates with status `solved`; no solver fallback or
final-motor cap occurred. Every control update used
`applied_motor_throttle` for all three `tau_achieved` axes. Full RPM validity
was `98.76%`, but the P95 absolute RPM-diagnostic minus throttle-derived force
difference was approximately `[2.12,1.80,2.11] N`, confirming the rollback.
Maximum tracker update time was `9.41 ms`, maximum motor magnitude was
`0.1522`, and the largest QP slack was `0.01155`.

The dominant interruption was the MPC-side visual distance gate: `612` of
`733` held-armed vision gaps were `forward_range`. During the same run the
read-only vision stream produced `542` target samples beyond the old
`0.85 m` maximum. After the operator confirmed that the vehicle stopped
following at long range and selected `1.4 m`, only
`experimental_auto.vision_gate_overrides.forward_range_m[1]` was changed from
`0.85` to `1.40 m`. The operator then selected a `0.30 m` experimental minimum
instead of `0.35 m`; the controller camera-forward constraints remain
`0.20--1.50 m`. NIS, depth-quality, motion-jump and reacquisition policies are
unchanged. No vision code or lower firmware changed.

## 2026-08-14: lateral-yaw A/B with yaw rate Kp 0.12

The visible yaw oscillation during sway was isolated to the lower attitude
loop/mixer before changing MPC speed or horizon. Only the lower yaw rate-loop
Kp was reduced from `0.15` to `0.12`; Ki remained `0.00005` per 1 kHz cycle,
Kd remained zero, the outer attitude gain remained `2.5`, and all MPC and
vision settings were held fixed. The rebuilt firmware was verified after
flashing, and the disarmed link checks before and after the run reported
`IDLE`, `armed=0`, zero requested/applied channels, zero RPM, and no rejected
commands.

The accepted trace is `experimental_auto_20260814_182757.jsonl`. It contains
`909/909` solved QP updates over `100.45 s`; median/P95/max tracker time was
`4.94/8.62/11.08 ms`. All three achieved-force axes used final applied motor
throttle, and all eight RPM bits were valid for `900/909` updates. There was no
final `0.50` motor scaling. The whole-run maximum motor magnitude of `0.1924`
and maximum slack of `0.4548` occurred during a later, deliberately excluded
distance/position excursion at camera position approximately
`[-0.43,0.14,1.34] m`, not during the clean lateral onset.

In the clean outbound onset (`30--39 s` from the first active update), onboard
camera x moved from about `+0.008 m` to a peak of `+0.089 m`; median camera
forward distance was `0.589 m` with range `0.573--0.629 m`. The right channel
peaked at `2.87%`. Absolute yaw rate median/P95/max was
`1.56/8.50/12.31 deg/s`, and yaw spanned `1.91 deg`. The four horizontal
motors made `68` zero/nonzero transitions in `8.85 s` (`7.69/s`), versus only
`1.41/s` in the preceding static baseline; at least one horizontal motor was
zero in `85.4%` of onset updates. Logged RPM shows that nonzero horizontal
outputs just above the current `1%` hard cutoff normally jump directly to
about `625--650 RPM`, confirming that the hard per-motor cutoff is a material
source of the visible switching.

Across the complete new trace, frames with `1--3%` absolute right request had
yaw-rate P95/max `5.66/9.83 deg/s`, compared with `6.84/16.95 deg/s` in the
preceding Kp `0.15` trace `experimental_auto_20260814_181604.jsonl`. For right
requests at or above `3%`, P95/max changed from `5.14/7.63` to
`2.87/9.25 deg/s`; the isolated valid return segment had P95/max
`5.08/7.14 deg/s`. Thus Kp `0.12` reduces the sustained/high-percent yaw
response but does not eliminate the onset chatter. Keep `0.12` provisionally;
the next single change should address horizontal-motor deadband switching,
not MPC translation weights. Only after that A/B should lateral speed or
horizon be changed.

The overhead camera was used only as auxiliary planar evidence. It confirmed
that the target motion was predominantly lateral relative to vehicle heading,
but its mapped displacement was roughly `0.4 m`, much larger than requested,
and vehicle-tag detections were sparse. Its underwater pose/yaw and apparent
depth are therefore not used as controller truth or as quantitative yaw
ground truth.

## 2026-08-14: model-1 equilibrium-relative force cost after forward H10 test

The translation-only trace `experimental_auto_20260814_205217.jsonl` used
`horizon=10`, `terminal_weight_scale=2`, and forward position/velocity/force/
delta-force weights `500/100/0.5/4.0`. The synchronized overhead bag
`top_camera_20260814_205217` independently observed target Tag 17 and vehicle
Tags 15/16. Its target forward-speed P95/maximum was approximately
`0.170/0.313 m/s`, versus `0.105/0.149 m/s` in the preceding H5 run, confirming
that the operator moved the target materially faster.

During repeated forward/reverse intervals, overhead target and vehicle mean
forward speeds were nevertheless close while the onboard forward error stayed
about `7--20 cm` in the direction of motion. The corresponding achieved-force
base was commonly `1--2.5 N`. A zero-relative-speed equilibrium solve with the
then-active absolute-force cost predicted forward offsets of `5.8/8.7/11.6/
14.4 cm` for `tau_base=1.0/1.5/2.0/2.5 N`, matching the observed lag. Increasing
only `position_weights[forward]` from `500` to `800` reduced the predicted
offset but did not remove it and increased requested correction; larger values
reached the `0.8 N/update` delta-force bound in recorded-state replay.

The operator clarified that model 1 is specifically intended to remove
constant-speed following lag. The effort reference was therefore changed from
zero absolute force to the fused zero-effective-input force:

```text
tau_eq,k = A1 tau_base,k + (I-A1) tau_h
J_force = sum_j (tau[j]-tau_eq,k)^T R (tau[j]-tau_eq,k)
```

The state equations, fixed-per-solve base, Fossen restoring force, absolute
force bounds, adjacent-force cost/limits, and fallback are unchanged. Offline
pure-model-1 solves now hold a nonzero base at zero position/velocity error;
the terminal forward error was below `0.02 cm` for `1--2.5 N` bases. The trace
now records `mpc_force_reference_frd_n`. No lower-controller firmware or vision
code changed. The next water test must keep `Q_forward=500` and the existing
fusion parameters fixed so this cost change is evaluated by itself.

## 2026-08-15: single-model-2 baseline versus adaptive fusion

The single-model-2 baseline trace `experimental_auto_20260815_163804.jsonl`
ran for about `111.5 s` with `fixed_model1_weight=[0,0,0]`. The subsequent
adaptive-fusion trace `experimental_auto_20260815_164204.jsonl` ran for about
`99.2 s` using the same vision source and MPC weights after removing the fixed
override. No firmware or vision code changed between the runs.

| Axis | Model 2 MAE/P95 | Fusion MAE/P95 | Fusion command saturation |
|---|---:|---:|---:|
| forward | `3.84/10.82 cm` | `7.56/22.18 cm` | `6.14%` |
| right | `2.41/5.65 cm` | `2.28/6.68 cm` | `0.23%` |
| down | `2.62/5.72 cm` | `2.64/9.46 cm` | `0.57%` |

Fusion forward weight had median `0.011` but P90 `0.862`, showing intermittent
model-1 excursions rather than a stable preference. The window filter itself
was active: median valid/rejected windows were `5/1` forward, `1/4` right and
`3/3` down. This run does not justify enabling the fusion setting unchanged
for forward control; the next offline candidate should reduce the weight
update rate (for example `0.35 -> 0.10`) while keeping the window rule,
threshold and MPC weights fixed.

## 2026-08-15: offline longer fusion prediction window candidate

The offline replay of five overhead-camera-paired traces compared the current
effective H2 history with completed H5 and H8 histories.  Forward candidate
separation in steady cruise increased from `0.803 mm` (P50) for the current
history to `2.835 mm` for H5 and `5.888 mm` for H8.  However, the fraction of
steady-cruise samples where model 1 had lower candidate MSE remained
`39.9%/39.3%/41.0%` for H2/H5/H8.  Longer history improves numerical
separation but does not by itself identify the better model.

The active experimental fusion block is therefore staged at the tested H5
candidate: `window=8`, `prediction_horizon=5`, unit prediction-step weights,
and staircase caps `[5,5,5,5,5,4,3,1]`.  This is an offline-selected
configuration only; no H5 hardware run has been started yet.

The H5 water run `experimental_auto_20260815_165500.jsonl` was then performed
with the same vision source and unchanged MPC costs. It ran for about `101.8 s`
and was operator-stopped after a vision gap. Forward/right/down MAE/P95 were
`8.45/21.8`, `4.24/13.79`, and `3.24/8.73 cm`; forward command saturation was
`10.82%`. The model-1 weight P90 reached `0.975/0.989/0.990` for the three
axes even though the medians stayed near `0.01`. The longer window therefore
increased candidate separation but amplified intermittent model-1 excursions;
H5/window8 is rejected for the next run. Return to the safer single-model-2
baseline or reduce the fusion update rate offline before another fusion test.

Following that result, only `weight_update_rate` was staged from `0.35` to
`0.10`; the long `H=5/window=8` history was intentionally left unchanged.
No new hardware run has been started with this rate yet.

The adaptive fusion preset was then changed to `initial_model1_weight=[0,0,0]`
while retaining `minimum_weight=0.01` and `fixed_model1_weight=null`. Thus the
runtime starts at the protected floor `0.01` on each axis and can still adapt;
it is not a permanent model-2 override.

## 2026-08-15: forward matched-motion baseline gate staged offline

The preceding gated EMA improved static holding but remained eligible on only
about `9%` of the dynamic forward samples in
`experimental_auto_20260815_175801.jsonl`, because it required
`|v_forward| <= 0.08 m/s`.  The active experimental configuration now adds a
separate forward-only gate: `|v|=0.05--0.20 m/s`, estimated acceleration at most
`0.20 m/s²`, and three consecutive same-sign confirmations.  Once confirmed it
uses the existing transient EMA rate `alpha=0.03`; static updates remain at
`alpha=0.01/0.03` under the original position/velocity gates.  Any sign change,
acceleration outside the bound, or position error above `0.20 m` resets the
dynamic confirmation streak and freezes the base.  Right/down and all MPC,
fusion, vision, firmware, and actuator settings are unchanged.  This is an
offline code/config change only; no hardware run has started with it.

The operator rejected the extra matched-motion threshold because freezing during
acceleration, reversal, or large error would further weaken dynamic following.
The active preset therefore disables `matched_motion_axis_enabled` and returns
to one position/velocity gated EMA on all axes, with a moderate speed-up from
`alpha=0.01/0.03` to `alpha=0.02/0.08` and the broad velocity gate widened from
`0.08` to `0.20 m/s`.  The `0.20 m` position gate is unchanged.  This change is
upper-controller configuration only; no hardware run has started with it.

## 2026-08-15: H15 run — forward-end visual depth instability

The follow-up H15 trace is
`calibration_logs/experimental_auto_20260815_210721.jsonl`. During the
straight-forward endpoint, raw visual frames `108493--108790` repeatedly
alternated between approximately `0.36` and `1.08 m` camera depth. The raw
vision stream also showed `0.15--0.25 m` jumps, approximately `0.20 s`
acquisition gaps, low depth quality, and NIS spikes above `200`. The interval
is excluded from offline MPC evaluation only; the raw trace is preserved and
the exclusion is recorded in
`calibration_logs/experimental_auto_20260815_210721.analysis.json`.

The whole trace had forward/right/down MAE `3.36/2.38/1.59 cm` and 3-D P95
`15.85 cm`. After excluding the visual interval, MAE became
`2.70/2.35/1.60 cm` and 3-D P95 `11.69 cm`; no MPC or firmware parameters were
changed. In the clean return segment (`frames 109800--110400`), raw depth
stayed around `0.58--0.63 m`, 3-D P95 was `7.79 cm`, and maximum applied motor
throttle was `0.097`. This confirms that the apparent forward-end oscillation
is primarily an intermittent visual measurement problem; the return appears
better because the measurement stream becomes continuous again.

The clean portions support good static/near-static holding, but every sustained
forward-depth excursion in this run overlaps degraded or jumping depth
measurements. Therefore a forward MPC lag cannot be identified independently
and no parameter change is justified. The experiment decision is to redesign
the next run: use `5--10 cm` forward/back steps, keep the target in the stable
`0.55--0.75 m` visual-depth band, hold each step for `10 s`, and repeat three
cycles. Vision quality will be classified offline only; the online controller
and visual code remain unchanged.

## 2026-08-15: H15 fusion run — visual endpoint exclusion and high-speed model check

The H15 upper-controller run is recorded in
`calibration_logs/experimental_auto_20260815_205021.jsonl`. The raw trace is
preserved. The second forward-run endpoint contains a visual depth/body-forward
jump: frames `84235--84424` move from the preceding approximately
`0.66/0.92 m` state to approximately `0.80--1.09/0.97--1.37 m` and then
return. That interval is excluded from offline error and model-selection
statistics only; the exclusion record is
`calibration_logs/experimental_auto_20260815_205021.analysis.json`.

After that exclusion, estimated FRD error MAE/P95 are
`3.44/10.87 cm` forward, `1.70/4.89 cm` right, and `1.68/3.89 cm` down.
The remaining larger forward peak is from other portions of the run and is not
automatically discarded. No raw JSONL rows were deleted.

In the final high-speed segment, frames `84880--84907` kept forward model-1
weight near `0.99` because its candidate MSE was lower. After the subsequent
velocity sign change, frames `84910--84940` reduced the weight to a median of
`0.053` (P90 `0.364`) while model 2 had lower candidate MSE. Thus the fusion
selector did distinguish this transition; the near-zero-acceleration plateau
itself still favored model 1. MPC was stopped by operator request; the vision
window and live error plot remain running.

## 2026-08-15: H15 redesigned small-step run — forward error remains dominant

The run is recorded in `calibration_logs/experimental_auto_20260815_212453.jsonl`;
the raw trace remains unchanged. The operator performed the planned axes, but
the depth also ranged from `0.331` to `1.076 m`, so larger hand-motion rows are
retained but are not treated as the clean `5--10 cm` baseline. The offline
summary is `calibration_logs/experimental_auto_20260815_212453.analysis.json`.

Across the trace, estimated FRD absolute error P95 was
`11.2/5.9/5.1 cm`. Restricting to the clean small-step rule (depth
`0.50--0.70 m`, camera `|x|,|y| <= 0.12 m`) gave `7.4/5.7/4.9 cm`; static
forward P95 was still about `6.3 cm`, while dynamic forward P95 was about
`17.5 cm`. Dynamic forward model-1 weight had median `0.023` and model 1 had
lower candidate MSE in only `17.1%` of dynamic samples, so the large dynamic
error is not caused by an indistinguishable fusion decision. Forward command
occasionally reached the `+/-0.20` channel/force boundary, but applied motor
throttle peaked at only `0.261`, well below the `0.50` motor limit. QP updates
were solved; vision acquisition P50/P95/max was `0.101/0.199/0.603 s`.

The next decision is a single parameter change, not a model or vision change:
test only forward `position_weight 800 -> 1000`, with horizon, fusion,
force/rate weights, vision and firmware unchanged. Repeat the small forward /
back step protocol at `0.55--0.75 m` and compare P95, overshoot, return time and
channel saturation. No parameter was changed by this run.

## 2026-08-15: forward weight 1000 run — high-speed segment is rate-limited, not motor-limited

The run is `calibration_logs/experimental_auto_20260815_215856.jsonl`; the
raw trace remains preserved and the detailed analysis is in
`calibration_logs/experimental_auto_20260815_215856.analysis.json`. Forward
position weight was the only staged change (`800 -> 1000`).

The clean small-step region (depth `0.50--0.70 m`, camera `|x|,|y| <= 0.12 m`)
had estimated error P95 `8.2/6.0/5.5 cm` (forward/right/down), so forward did
not yet establish a reliable improvement to the 5 cm target. The later
high-speed samples (`|v_forward| >= 0.10 m/s`) had forward error P50/P95/max
`6.7/25.1/60.7 cm`. Model-1 weight median was only `0.03` on forward, so
model 2 was selected during the fast motion.

This was not final-motor saturation: maximum applied throttle was `0.274`,
well below `0.50`. However, the first planned forward force increment hit the
`+/-0.80 N` rate bound in about `65%` of high-speed samples; forward command
also occasionally reached the experimental `+/-0.20` channel envelope. The
main high-speed bottleneck is therefore force-rate/actuator dynamics, not a
lack of position weight or final motor authority. A later fast interval also
contained substantial lateral/vertical motion, so it is marked as a mixed
high-speed segment rather than a pure forward benchmark. The pool-top MP4 was
still open and lacked a final `moov` atom, so quantitative frame extraction was
not possible without closing the camera recorder.

Next decision: test only forward `delta_force_min/max +/-0.80 -> +/-1.00 N`,
keep position weight `1000` and all other settings fixed, then repeat a
straight high-speed forward/back segment using the overhead view to constrain
lateral/vertical motion. No parameter was changed by this analysis.

## 2026-08-15: AUTO startup diagnosis after forward-weight change

After staging forward `position_weight 800 -> 1000`, three launch attempts
(`experimental_auto_20260815_213656.jsonl`, `214232.jsonl` and
`215120.jsonl`) produced only a `start` record and no `control_update`. The
default configuration pointed to the old 2026-08-13 vision JSONL, whose tail
was not advancing, so `measurement_fresh` never became true. The guarded AUTO
therefore correctly stayed in `DISARMED_HANDSHAKE` and sent no MPC command.

The UDP transport was not the root cause: the experiment uses
`remote_host=192.168.0.2` (reachable), while `host=192.168.138.2` is the old
TCP field and must not be used for this check. A disarmed link test confirmed
fresh telemetry, execution feedback and `mpc_direct=1`; it also records the
lower-controller yaw mode, which must be `LOCAL_HOLD` (`telemetry_yaw_direct=0`)
for the translation model.

The retry using the live vision path
`real_device_20260815_200705/pipeline_results.jsonl` entered `ACTIVE` and
generated 192 solved control updates in
`experimental_auto_20260815_215559.jsonl` before the operator stopped it.
This short run is not a tuning result; it only proves that the corrected
vision path and lower-controller mode allow the staged parameter to run.

## 2026-08-15: forward-rate run — control solved, confirmation watchdog tripped

The run is `calibration_logs/experimental_auto_20260815_221903.jsonl`.
It entered `ACTIVE` after the live vision path became valid and produced 1460
control updates over about 158 s. The final event was
`armed command confirmation stale`; this is a communication/acknowledgement
safety stop, not a vision rejection, QP fallback, or motor saturation.

Immediately before the stop, MPC status was `solved`, the maximum slack was
about `8e-8`, tracker computation was about `9.1 ms`, and the largest final
motor throttle in the complete run was about `0.362`, below the `0.45--0.50`
motor-limit diagnostic threshold. The last requested channels were only
`[-0.079, +0.046, +0.009]`, so the stop was not caused by the experimental
translation envelope either. The last control record was followed by the
watchdog stop after roughly `0.24 s`, matching the configured
`confirmation_max_age_sec=0.25` boundary.

A subsequent 4 s disarmed diagnostic sending `yaw_direct=0` received 78/79
fresh telemetry samples, with accepted commands and no reject flags. This
supports an intermittent telemetry/command-echo gap (or a host receive delay)
under the continuous 20 Hz AUTO stream, rather than a persistent lower
controller fault. Do not tune MPC from this run. Before the next long test,
collect confirmation round-trip/telemetry-age diagnostics; if the gap recurs,
adjust the watchdog margin or transport handling as a communication fix and
repeat the same motion, keeping MPC parameters unchanged.

The useful control portion of this same run is also informative after
separating obvious depth bursts. In the raw vision stream, frame `231199`
has `depth_nis=572.8` and frame `231223` has `depth_nis=325.3`; similar bursts
occur at `231343` and `231355`. The current experimental gate rejects those
individual samples but then accepts several following low-NIS samples while
the depth track is still wrong, using `clamp_implausible_steps`. Around
`231199--231244`, the camera depth walks from about `0.624` to `1.35 m` and
back to `0.58 m`, producing a spurious body-forward error up to about `0.74 m`.
Those rows must be excluded from MPC performance scoring; they are a vision
burst, not a vehicle response.

After excluding the two obvious bursts (`231199--231244` and
`231343--231367`), low-speed samples (`|v|<0.03 m/s`) have forward error
median/P95 about `2.8/8.8 cm`. Forward dynamic samples (`|v_forward|>0.10
m/s`) have error median/P95 about `6.6/25.2 cm`, while the forward model-1
weight has median about `0.02` (model 2 is correctly selected). In contrast,
the low-speed model-1 median is about `0.68`; the fusion distinction is
therefore visible in this run rather than being lost in noise.

The dynamic bottleneck is the forward force-rate constraint: the first
planned forward increment reaches the current `+/-1.00 N` bound in about
`56%` of samples with `|v_forward|>0.10 m/s`. The forward channel reaches its
`+/-0.20` envelope in only about `8%` of those samples, and clean dynamic
motor throttle peaks at about `0.289`, far below the final motor limit. Thus
the remaining dynamic lag is consistent with force-rate/actuator dynamics,
not model confusion or final-motor saturation. The next tuning experiment
should change only the forward force-rate limit (or, preferably, first
validate the actuator time constant) and keep position weights unchanged;
the visual-burst rows must remain marked invalid for scoring.

## 2026-08-15: depth continuity gate for the next tuning run

To prevent the sustained false-depth sequence observed in the previous run,
the experimental host-side vision gate now keeps the absolute range at
`0.30--1.40 m` but changes the motion handling to:

- `clamp_implausible_steps=false`: an implausible sample is rejected and the
  last trusted measurement is retained; it is not converted into a gradual
  fake trajectory;
- `jump_margin_m=0.20`, `max_speed_m_s=1.0`, and a hard
  `max_step_m=0.30 m`, so the allowed spatial jump is at most about `0.30 m`
  per visual update and never expands with a long sample gap;
- `max_inter_sample_gap_s=1.0`, allowing the last trusted depth to remain the
  reference while a short false burst is rejected.

The previous JSONL replay rejects the `231199--231220` false-depth run and
does not feed its `0.74 m` apparent forward error to MPC; the measurement is
accepted again when the depth returns near the last trusted value. This is an
MPC-side input gate change only; the tracking-depth code and firmware are
unchanged. The staged MPC weights and forward `+/-1.00 N` force-rate limit
remain unchanged for the next comparison run.

## 2026-08-15: repeated AUTO startup stall caused by stale vision JSONL

The launch `experimental_auto_20260815_233324.jsonl` contained only its
`start` record. The MPC preflight output still pointed to the old
`finesub_v4pro1_camera_cal_20260812_run02/pipeline_results.jsonl`, whose tail
was not advancing. The active vision producer was instead writing
`real_device_20260815_200705/pipeline_results.jsonl`. Because the guarded
runtime tails the selected file from EOF, no fresh measurement could arrive;
it correctly stayed in `DISARMED_HANDSHAKE` and never entered `ACTIVE`.

This was a recurrence of the startup fault already documented above, not a
new MPC or SMC behavior difference. The lower link was independently healthy:
`link_diag_20260815_2336.csv` recorded 200 disarmed rows, 197 unique fresh
telemetry sequences, 198 accepted command confirmations, and no armed or
nonzero command. The operator stopped the stalled AUTO before any control
update.

Before every real-device launch, identify the currently growing
`pipeline_results.jsonl`, pass it explicitly with `--vision-jsonl`, and verify
that its tail timestamp advances. Do not rely on the dated path in
`finesub_v4pro1_mpc.json`; it is only a historical default. The preflight must
print the same live path before adding `--execute`.

## 2026-08-15: overhead target-speed panel data coverage and SQLite lock

The target-speed panel was blank for most of
`experimental_auto_20260815_234505.jsonl` because the paired pool-top bag
`top_camera_live_20260815_234411` covered only 15.4 s, while the MPC trace
covered 96.7 s; only two control samples overlapped the bag time range. The
bag recorder also stopped with SQLite `database is locked` while the live
plot was reading the writable database.

The plot reader now opens a read-only immutable SQLite snapshot, so it cannot
hold a writer lock. A speed value is still only available when the status
stream contains at least 20 simultaneous target Tag 17 and vehicle Tag 15/16
detections; if Tag 17 is not visible, an empty speed panel is expected. For a
live experiment, keep the pool-top recorder running for the complete MPC run
and keep the target tag in view.

## 2026-08-18: long dual-model AUTO run recorded before pure model-1 isolation

The experiment trace was preserved at
`calibration_logs/experimental_auto_20260818_030626_realtime.jsonl`. The
active camera source was explicitly passed as
`/data/Zhouyuh_workspace/tracking_depth/depth-estimation-dev/output/finesub_mpc_20260818_020915/pipeline_results.jsonl`;
the dated path in the JSON configuration was stale and must not be reused.

The run entered ACTIVE after the disarmed handshake and lasted about 777 s,
with 7,236 solved control updates. There were 59 short visual-gap hold
records and four hold episodes; no QP failure occurred. Forward requested
channel reached both `-0.20` and `+0.20`; right ranged approximately
`-0.130..+0.153`, and down `-0.113..+0.094`. The maximum final motor
throttle was `0.349`, below the `0.45--0.50` reporting threshold. Telemetry
yaw stayed near `1.48..1.60 rad`; this run did not show a large 90-degree yaw
excursion. Maximum reported QP slack was `0.0163`, with P95 about `2.0e-5`.

After stopping, the next isolation run fixes `fixed_model1_weight` to
`[1.0, 1.0, 1.0]` in the translation fusion configuration. This disables
online fusion updates and uses model 1 on all three FRD axes; model 2 remains
available only for comparison diagnostics.

## 2026-08-18: pure model-1 isolation run

Trace: `calibration_logs/experimental_auto_20260818_032311_model1fixed_realtime.jsonl`.
The run lasted about 500 s and produced 4,625 solved control updates. The
logged model-1 weight was `[1.0, 1.0, 1.0]` on every update. Forward reached
both channel limits (`-0.20` and `+0.20`) 57 times; right ranged
`-0.174..+0.154`, down `-0.103..+0.096`. Maximum final motor throttle was
`0.343`, still below the `0.45--0.50` reporting threshold. QP status was
`solved` for all updates. Vision-gap hold records were frequent (288 records,
four hold episodes), with measurement-age P95 about `0.111 s` and maximum
about `0.130 s`. Yaw-delta P95 was about `0.40°` and yaw-rate P95 about
`4.14°/s`; no large yaw excursion was observed.

For the next isolation run, `fixed_model1_weight` is set to `[0.0, 0.0, 0.0]`,
so model 2 is used on all three FRD axes and online fusion updates remain
disabled.

## 2026-08-18: pure model-2 isolation run

Trace: `calibration_logs/experimental_auto_20260818_033331_model2fixed_realtime.jsonl`.
The run lasted about `210.3 s` and produced `1,906` solved control updates.
The logged model-1 weight was `[0.0, 0.0, 0.0]` throughout, so model 2 was used
on all three FRD axes and online fusion updates were disabled. Forward ranged
from `-0.1966` to `+0.20`, touching the positive channel limit only twice;
right ranged `-0.133..+0.102`, and down `-0.062..+0.148`. The maximum final
motor throttle was `0.327`, well below the `0.45--0.50` diagnostic threshold.
All QP solves were `solved`. Vision measurement age had P95 `0.111 s` and
maximum `0.140 s`; yaw increment P95 was `0.426°` and yaw-rate P95 was
`4.44°/s`. No sustained large yaw excursion was observed.

After this isolation run, `fixed_model1_weight` is restored to `null` so the
configured adaptive fusion (`window=8`, prediction horizon `5`,
`weight_update_rate=0.10`, initial model-1 weight `[0,0,0]`) is active again.

## 2026-08-18: five adaptive-fusion forward windows versus fixed-model baselines

Offline comparison output:
`calibration_logs/mpc_fusion_vs_fixed_200cm_20260818.png` and
`calibration_logs/mpc_fusion_vs_fixed_200cm_20260818.json`.
The five recent adaptive-fusion traces (`034641`, `034921`, `035124`,
`035310`, and `035442`) use their initial 20 s windows.  The fixed model-1
trace uses `experimental_auto_20260818_032311_model1fixed_realtime.jsonl` at
`60--80 s`; the fixed model-2 trace uses
`experimental_auto_20260818_033331_model2fixed_realtime.jsonl` at `15--35 s`.
The comparison uses the Kalman estimated forward error and the mean of its
absolute value in 100 ms bins.  Phase boundaries are 0--5 s acceleration,
5--10 s constant speed, 10--15 s stop, and 15--20 s post-stop observation.

Mean absolute error by phase (cm):

```text
                         acceleration   constant   stop   post-stop
adaptive fusion (5-run mean)   5.01         5.33     3.03     4.66
fixed model 1                  1.76         4.33    10.05     5.86
fixed model 2                  5.70         9.25     9.46     4.14
```

The selected fixed-model-1 stop window reaches `19.2 cm` absolute forward
error, while fixed model 2 has a `+9.25 cm` signed mean error in its selected
constant-speed phase.  The adaptive-fusion mean is lower in the constant and
stop phases than both fixed baselines, but its five-run spread is substantial.
The additional post-stop interval is retained specifically to show the
overshoot/recovery trend.  No synchronized pool-top bag was recorded for these
exact 20260818 runs, so the historical segmentation is based on MPC timing and
forward-leg error/force signatures; the selected windows and definitions are
preserved in the JSON summary.

## 2026-08-18: comparison revised to include post-stop observation

The previous separate static model-2 panel has been removed from the comparison
figure and JSON summary.  The current output keeps one common dynamic window
for all three cases: approximately `15 s` of forward motion followed by `5 s`
of post-stop observation (`0--20 s` total).  The four phase boundaries are
`0--5 s` acceleration, `5--10 s` constant speed, `10--15 s` stop, and
`15--20 s` post-stop observation.  This makes the stop overshoot and recovery
visible without presenting model 2 as a separate static experiment.

Current mean absolute error by phase (cm):

```text
                         acceleration   constant   stop   post-stop
adaptive fusion (5-run mean)   5.01         5.33     3.03     4.66
fixed model 1                  1.76         4.33    10.05     5.86
fixed model 2                  5.70         9.25     9.46     4.14
```

The five fusion traces and both fixed-model traces all contain at least 20 s
in their selected windows, so the full 5 s post-stop interval is retained.  If
a future trace ends earlier, the post-stop interval may be shortened to the
available samples rather than discarding the entire run.  No synchronized
pool-top bag was recorded for these exact runs; segmentation remains based on
MPC timing and forward error/force signatures.
