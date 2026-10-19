# Methodology

This document describes exactly what the demo computes. Items marked **[paper]** follow
arXiv:2609.12184v2. Items marked **[demo choice]** are this repository's own choices, made where the
paper is silent or where the simplified model requires them.

## 1. Synthetic device (`src/agentic_tcad_demo/device.py`)

The model is a closed-form transfer-curve model of a bottom-gate TFT. It is **not** a device-equation
solver: there is no Poisson equation, no drift-diffusion, no 2D geometry, and no trap dynamics. The
geometry follows the paper's Table III: W = 5 µm, L = 1 or 2 µm, channel 3.5 nm, gate dielectric 7 nm
**[paper]**. The dielectric constant (κ = 20), electron affinity (4.30 eV), and contact constants are
illustrative **[demo choice]**.

| Quantity | Expression |
|---|---|
| Oxide capacitance | C<sub>ox</sub> = ε<sub>0</sub>κ / t<sub>ox</sub> |
| Threshold voltage | V<sub>T</sub> = (φ<sub>G</sub> − χ) − qQ<sub>f</sub>/C<sub>ox</sub> (Q<sub>f</sub> only if `fixed_charge` is active) |
| Subthreshold ideality | n = 1 + C<sub>it0</sub>/C<sub>ox</sub> + qN<sub>tail</sub>t<sub>ch</sub>/C<sub>ox</sub> (tail term only if `tail_states` is active) |
| Intrinsic current (EKV interpolation) | I = 2nμC<sub>ox</sub>(W/L)V<sub>t</sub>²[F(V<sub>GS</sub>−V<sub>T</sub>) − F(V<sub>GS</sub>−V<sub>T</sub>−nV<sub>DS</sub>)], F(x) = ln²(1+e<sup>x/2nV<sub>t</sub></sup>) |
| Contact resistance (S + D) | R<sub>c</sub> = 2ρ<sub>c0</sub>/W · exp(φ<sub>B</sub>/(m kT/q)), with φ<sub>B</sub> = softplus(φ<sub>SD</sub> − χ) |
| Self-consistency | I = I<sub>int</sub>(V<sub>G</sub> − IR<sub>c</sub>/2, V<sub>D</sub> − IR<sub>c</sub>), solved by vectorized bisection |

Because R<sub>c</sub> does not scale with L while the channel resistance does, channel length changes the
balance between contact- and channel-limited current. The L = 2 µm test therefore exercises the model.

**Convergence.** A simulation reports `converged = False` in three cases:
- the parameters are non-finite or μ0 ≤ 0
- the contact barrier exceeds the validity limit of the expression (0.8 eV)
- the self-consistent solve misses its tolerance

The workspace then tries one predefined recovery: the same solve with twice the iterations. If that also
fails, the run is recorded as a numerical failure and excluded **[paper: recovery, then exclusion. The
specific recovery step is a demo choice]**.

**Synthetic truth ≠ calibration model [demo choice].** The "measured" curves come from the same model
with all components active plus two effects the calibration model never has:
- a gate-dependent mobility μ<sub>0</sub>(1 + V<sub>ov</sub>/1 V)<sup>0.25</sup>, loosely in the spirit of percolation transport in amorphous oxides
- a drain-induced threshold shift of 30 mV/V × (1 µm / L)

Noise is 3% lognormal multiplicative noise plus a 2 pA additive floor. Every condition draws from its own
child stream of one `numpy.random.SeedSequence(2609)`. This prevents an "inverse crime": a perfect fit
cannot exist, and the held-out errors reflect model-form error.

The truth parameters are φ<sub>SD</sub> 4.35 eV, φ<sub>G</sub> 4.30 eV, μ<sub>0</sub> 18 cm²/Vs,
Q<sub>f</sub> 6×10¹² cm⁻², and N<sub>tail</sub> 2×10¹⁹ cm⁻³eV⁻¹. They are illustrative and chosen to
differ from the paper's calibrated values. They are stored in `summary.json` for reference only and are
never visible to the controller.

## 2. Metrics and objective (`metrics.py`, `objective.py`)

- V<sub>th</sub>: V<sub>G</sub> at \|I<sub>D</sub>\| = 100 nA **[paper]**. Linear interpolation in log I **[demo choice]**.
- SS: average inverse slope between 0.1 nA and 100 nA, i.e. ΔV<sub>G</sub>/3 decades **[paper]**. The unit of the lower bound is read as nA.
- I<sub>on</sub>: maximum \|I<sub>D</sub>\| **[paper]**.
- E<sub>log</sub>: RMSE of log<sub>10</sub>\|I<sub>D</sub>\|, excluding measured points below 10 pA **[paper]**.
- E<sub>lin</sub>: RMSE of \|I<sub>D</sub>\| within one decade of I<sub>on</sub> **[paper]**, relative to the measured I<sub>on</sub> **[demo choice]**.
- J: Eq. (1) with weights w<sub>log</sub> 0.25, w<sub>lin</sub> 0.10, w<sub>th</sub> 1.00, w<sub>on</sub> 0.20, w<sub>SS</sub> 0.25, averaged over the calibration biases V<sub>D</sub> ∈ {0.05, 1.0} V **[paper]**.
- Normalization a<sub>i</sub> **[demo choice]**: 0.1 dec (log), 0.1 × I<sub>on</sub> (lin), 0.1 V (V<sub>th</sub>), 0.1 dec (I<sub>on</sub>), 10 mV/dec (SS). The paper says each term is normalized by a fixed factor but does not give the values. **Absolute J values are therefore not comparable with the paper's.**
- The V<sub>G</sub> sweep runs from −1.5 to 2.5 V in 20 mV steps **[demo choice]**. A 50 mV grid created interpolation artifacts in the metric sensitivities.

## 3. Sensitivity and bounded update (`sensitivity.py`, `update.py`)

**Coordinates.** Each parameter maps to a normalized coordinate x ∈ [0, 1]. The densities N<sub>tail</sub>
and μ<sub>0</sub> use a logarithmic map **[demo choice]**.

**Residuals.** The residual vector r (Eq. 2) stacks three blocks:
- pointwise log-current residuals
- pointwise linear-current residuals, each block weighted w/N
- the three metric residuals (V<sub>th</sub>, log I<sub>on</sub>, SS)

**OFAT.** Each parameter of the selected group is moved by 0.02 in x, one at a time. The step goes
backwards if it would cross a bound. This gives S<sub>P</sub> = ∂r/∂x<sub>P</sub> by forward differences
**[paper: OFAT. The step size is a demo choice]**. A probe that fails to converge contributes a zero
column, so it can never create a descent direction.

**Update.** Eq. (3) is solved as a small convex QP:

  min ½‖W<sup>½</sup>(r + S<sub>P</sub>s)‖² subject to 0 ≤ x + s ≤ 1 and ‖s‖₂ ≤ Δ.

The solver first runs box-constrained least squares (BVLS). If the result lies outside the trust ball,
SLSQP adds the ball constraint **[demo choice]**. The default trust radius is Δ = 0.4. After a rejected
trial, Δ ← ½ min(Δ, ‖s‖), with at most two reductions per group per cycle **[demo choice]**.

Eq. (3) minimizes a squared surrogate of the absolute-value terms in Eq. (1). Acceptance is always
decided on J itself.

## 4. Controller and agent (`controller.py`, `tools.py`, `llm.py`)

Each cycle runs these steps:
1. **Observe.** Simulate the calibration conditions, compute J and its weighted terms, and identify the
   dominant term.
2. **Diagnose.** Map the dominant term to physically expected parameter groups or components, using
   `DIAGNOSIS_RULES`. The diagnosis is logged as evidence.
3. **Rank.** Run OFAT and the Gauss–Newton model on every available group, and log the predicted gain of
   each.
4. **Choose.** The planner picks one legal action. The rule planner applies four rules in order:
   - activate an eligible component that addresses the dominant residual
   - otherwise update the untried group with the largest predicted gain
   - otherwise, when no untried group predicts a gain, activate any eligible component
   - otherwise stop

   In the default run, `fixed_charge` is activated through the third rule. The dominant term is the
   log-current error, the only group with a predicted gain (contact) was already tried and rejected in
   that cycle, and the V<sub>th</sub> term carries 34% of J.
5. **Trial.** Propose the bounded step, check the bounds, and simulate.
6. **Decide.** Accept only if every calibration simulation converges and J<sub>new</sub> < J<sub>old</sub>
   **[paper]**. Otherwise log the rejection and its reason, shrink Δ, and finally try another group.

**Model activation [paper: concept. Thresholds are demo choices].** A component becomes *eligible* only
when all three conditions hold:
1. Its residual term carries ≥ 15% of J.
2. The term has done so at the start of 2 consecutive cycles, with a bounded parameter update in
   between, or no group predicts any gain at all. This does not require every parameter group to have
   been tried. In the default run, `tail_states` is activated while `fixed_charge` and `electrostatics` still predict
   gains.
3. Its trigger holds:
   - `tail_states` ("insensitive"): the best first-order reach of any single active parameter on the SS
     residual, max<sub>j</sub> |S<sub>SS,j</sub><sup>T</sup> r<sub>SS</sub>| / ‖r<sub>SS</sub>‖, is below
     0.5 × ‖r<sub>SS</sub>‖. In words, to first order, moving any one active parameter across its full
     range would remove less than half of the SS mismatch. Only fully converged OFAT rankings count:
     a failed probe is never treated as evidence of insensitivity.
   - `fixed_charge` ("railed"): the gate work function sits at an approved bound, and the residual
     gradient pushes it further into that bound.

Activation then requires an approval record. The default policy file approves both components, and
`--interactive` asks on the terminal instead. The component's parameter group is updated in the same
trial, and the component is kept only if J decreases.

**Stopping.** When two accepted updates in a row improve J by less than 1%, parameter updates are marked
"stalled". From then on only an activation or a stop is legal. A successful activation re-opens
parameter updates. The run stops in either of two cases:
- the planner stops, because no group predicts a gain or updates are stalled, and no component is eligible
- 25 accepted updates are reached

Seed 2609 stops by stalling. Six of seeds 2600–2609 reach the cap while still making small gains.

**LLM planner.** The LLM receives the same observation JSON as the rule planner. That includes:
- the current parameter values
- J and its terms
- the OFAT gains
- the legal actions and eligible components

It returns one JSON action naming an action kind and a target. It never proposes numeric parameter
values, because steps always come from Eq. (3). These answers count as planner errors:
- malformed output
- an unknown action
- a non-string target
- an action that is not legal in the current state (for example, an activation before the component is eligible)

Three planner errors stop the run. There is no fallback to the rule planner.

## 5. Evaluation (`evaluate.py`)

The final parameters are frozen and hashed. Only simulation and scoring tools are used, and tests assert
that no update function is called. The calibration split (L = 1 µm; V<sub>D</sub> = 0.05, 1 V) and the
held-out splits are evaluated and reported separately, for both the initial and the final model:
- held-out drain bias: L = 1 µm; V<sub>D</sub> = 0.1, 0.2, 0.5 V
- held-out length: L = 2 µm; V<sub>D</sub> = 0.05, 1 V

The paper's drain-bias set also contains 0.05 and 1.0 V. Here those two are shown as calibration
conditions, so the held-out drain set is strictly disjoint.

## 6. Known limitations and observations

- **Identifiability.** φ<sub>G</sub> and Q<sub>f</sub> enter only through V<sub>T</sub>, so only their
  combination is identified. In the default run they end at 4.50 eV and 9.2×10¹² cm⁻² (truth 4.30 eV and
  6×10¹² cm⁻²), with the correct V<sub>T</sub>. The paper makes the same point: fitted variables are
  calibration-effective descriptors, not extractions.
- **μ<sub>0</sub> compensation.** μ<sub>0</sub> converges to about 24 cm²/Vs (truth 18) because it
  absorbs the gate-dependent mobility that the calibration model lacks.
- **Close recovery.** φ<sub>SD</sub> and N<sub>tail</sub> are recovered closely (4.348 eV and
  2.0×10¹⁹), because their effects are distinct.
- **Robustness to Δ.** The result is insensitive to the trust radius in the range 0.3–0.7: final J was
  0.40–0.43, both components were activated, and max held-out |ΔV<sub>th</sub>| was 15.9–17.2 mV. The
  number of accepted updates (17–24) and rejected trials (0–42) changes with Δ.
  `tests/test_workflow.py::test_outcome_is_robust_to_the_trust_radius` loosely checks this (both
  activations, J reduced by more than 20×). The exact README values are pinned by
  `test_readme_numbers_match_a_fresh_run`.
- **Seeds.** Over noise seeds 2600–2609, final J is 0.34–0.43, both components are always activated, and
  max held-out |ΔV<sub>th</sub>| is 17.0–22.9 mV. Seed 2609, used in the README, is on the favorable side.
- **Controls.**
  - With the noise switched off, V<sub>th</sub> errors stay at 14–19 mV. With the extra truth physics
    removed, they fall to 2.6–3.5 mV. The residual is therefore model-form error.
  - The candidate components use the truth's own functional forms. If the truth contains none, one, or
    both of them, exactly those are activated (`test_activation_gate_is_selective`).
- **Null-space drift.** Any strict J decrease is accepted. The last accepted updates of the default run
  move φ<sub>G</sub> and Q<sub>f</sub> together along their unidentifiable direction for relative gains
  below 0.2%. They count among the 17 accepted updates.
- **Failure path.** The default run has no failed simulation, because the contact-barrier validity limit
  cannot be reached inside the bounds. The recover-then-exclude path is exercised only by
  `tests/test_acceptance.py` with an injected failure.
- **Scope.** The demo does not implement RAG, the paper's Gaussian/tail DOS models, the 4% W process
  test, or any Sentaurus feature.
