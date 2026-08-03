# Tendon-Compliance MocoInverse Debug Summary

## Goal

Get `run_mocoinverse.py` to solve the Walk05 MocoInverse problem with
`ModOpUseImplicitTendonComplianceDynamicsDGF()` enabled.

## Stable Base Context

- Trial: `BAA01_Baseline_Walk05`
- Model: `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/scaled_moco.osim`
- IK: `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/BAA01_Baseline_Walk05_ik.mot`
- External loads: `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/BAA01_Baseline_Walk05_fp_setup.xml`
- Warm-start source: `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/MocoInverse_Walk05/moco_inverse_Walk05.sto`
- IK preprocessing: 15 Hz low-pass filter, degrees-to-radians, absolute state names

## What Was Tried

### 1. Direct tendon-compliance enablement

- Change: switched from `ModOpIgnoreTendonCompliance()` to
  `ModOpUseImplicitTendonComplianceDynamicsDGF()`.
- Result: did not reach IPOPT within the initial timed foreground run.
- Interpretation: warm start from the rigid-tendon solution was not sufficient
  by itself.

### 2. Warm-start from the rigid-tendon inverse solution

- Change: initialized the compliant-tendon solver with the successful rigid-
  tendon MocoInverse solution.
- Result: still stalled before useful solver progress.
- Interpretation: overlapping states/controls were helpful in principle, but
  tendon states remained under-specified.

### 3. Solver reset ordering fix

- Change: changed the setup order to:
  `initialize() -> resetProblem(problem) -> createGuess() -> setGuess()`.
- Result: eliminated the earlier hard crash path in the guess setup.
- Interpretation: `resetProblem(problem)` needed to happen before creating the
  solver guess.

### 4. Flat tendon-state initialization

- Change: initialized `normalized_tendon_force` to `0.1` for all compliant
  muscles in the guess.
- Result: the compliant-tendon solve began entering IPOPT instead of dying
  during setup.
- Interpretation: tendon states must be initialized explicitly; a rigid-tendon
  state/control warm start alone is incomplete.

### 5. Additional stabilization goals/settings

- Changes:
  - added `MocoInitialVelocityEquilibriumDGFGoal`
  - enabled `minimize_implicit_auxiliary_derivatives`
  - used `implicit_auxiliary_derivatives_weight = 0.01`
- Result: this produced the first robust IPOPT entry for the compliant-tendon
  formulation.
- Interpretation: the compliant-tendon problem is runnable with these settings.

### 6. 20 ms mesh run

- Configuration: 20 ms mesh, tight tolerances, rigid-tendon warm start,
  tendon-force initialization.
- Result: ran deeply into IPOPT (hundreds of iterations) but stalled with
  primal infeasibility around `0.20–0.25`.
- Interpretation: setup is valid, but convergence is incomplete.

### 7. 10 ms mesh run

- Configuration: same as above, but mesh interval reduced to `0.01`.
- Result: improved early infeasibility significantly, reaching approximately
  `inf_pr ~ 0.15–0.17` early, but still failed to converge cleanly.
- Interpretation: this is the best-performing base configuration so far.

### 8. Stronger implicit auxiliary derivative weight

- Change: `implicit_auxiliary_derivatives_weight = 0.1`.
- Result: worse early IPOPT behavior than the `0.01` setting.
- Interpretation: heavier tendon-derivative smoothing is not the right next
  lever for this problem.

### 9. Loose global tolerances

- Changes:
  - convergence tolerance `1e-2`
  - constraint tolerance `1e-2`
- Result: did not improve the early search path and did not produce a clean
  solution.
- Interpretation: this is not primarily a termination-tolerance problem.

### 10. 5 ms mesh run

- Configuration: mesh interval reduced to `0.005` on the same stabilized base.
- Result: reached IPOPT, but early infeasibility was worse than the 10 ms case.
- Interpretation: finer mesh beyond 10 ms is not helping this model.

### 11. Central finite differences

- Change: set `optim_finite_difference_scheme = "central"`.
- Result: early IPOPT trajectory was effectively unchanged from the plain 10 ms
  run.
- Interpretation: derivative-accuracy change alone does not break the plateau.

### 12. Implicit multibody acceleration regularization

- Changes:
  - `minimize_implicit_multibody_accelerations = True`
  - `implicit_multibody_accelerations_weight = 0.1`
- Result: early IPOPT trajectory again matched the plain 10 ms branch closely.
- Interpretation: this regularizer does not appear to be the missing piece.

### 13. SO-informed tendon-force guess

- Change: replaced the flat `normalized_tendon_force = 0.1` initialization with
  a per-muscle, time-varying guess derived from
  `SO_physiology/scaled_scaled_StaticOptimization_force.sto`, normalized by each
  muscle's `max_isometric_force` and interpolated onto the Moco guess time grid.
- Result: startup improved materially. The 10 ms compliant-tendon solve still
  started around `inf_pr ~ 6.7`, but it reached roughly `0.148` early instead of
  flattening around `0.30`.
- Interpretation: the SO-derived tendon-force guess is better than the flat
  guess, but it does not solve the later infeasibility plateau.

### 14. Softened initial velocity equilibrium goal

- Change: changed `MocoInitialVelocityEquilibriumDGFGoal` from the default/hard
  setup into a low-weight cost.
- Result: this dramatically improved the initial feasible neighborhood. The
  solve started near `inf_pr ~ 0.443` instead of `~6.7` and still reached about
  `0.146–0.147` early.
- Interpretation: the equilibrium goal strongly affected startup, but not the
  later plateau.

### 15. Removed initial velocity equilibrium goal entirely

- Change: removed `MocoInitialVelocityEquilibriumDGFGoal` completely while
  keeping the SO-informed tendon-force guess.
- Result: startup stayed good and early infeasibility still reached about
  `0.146–0.148`, but the solve again drifted back into the same
  `~0.16–0.17` band.
- Interpretation: the equilibrium goal was not the main late-stage blocker.

### 16. Reduced and removed implicit auxiliary derivative regularization

- Changes:
  - `implicit_auxiliary_derivatives_weight = 0.001`
  - `implicit_auxiliary_derivatives_weight = 0.0`
- Result: neither change materially improved the later IPOPT path. All three
  settings (`0.01`, `0.001`, `0.0`) produced essentially the same early drop to
  `~0.147–0.148` and the same later plateau in the `~0.16–0.17` range.
- Interpretation: auxiliary-derivative regularization is not the primary late-
  stage blocker.

### 17. Staged 20 ms -> 10 ms compliant-tendon continuation

- Change: added a coarse compliant-tendon stage on a 20 ms mesh and attempted
  to warm-start the 10 ms stage from the compliant-tendon stage-1 solution.
- Result: the staged run did not help. Stage 1 itself remained trapped and did
  not provide a successful continuation path to Stage 2.
- Interpretation: this is not primarily a 10 ms transcription issue.

### 18. Per-muscle tendon-compliance diagnostic

- Change: created a standalone diagnostic script,
  `/home/hudson/Downloads/CMC_Runs/diagnose_tendon_compliance_muscles.py`, to
  scan muscles over Walk05 and rank suspect muscles based on geometry margin and
  safe tendon-force capacity.
- Result: no hard geometry violations or tendon buckling appeared, but a broad
  subset of muscles looked risky rather than a single isolated outlier.
- Interpretation: the issue appears broader than one rogue tendon-compliance
  muscle.

### 19. Selective ignore-tendon-compliance trial

- Change: built a selective rigid-tendon subset model and tested a first set of
  suspect muscles (`L_GS`, bilateral `GMa`, `IP`, `AL`, `STp`).
- Result: after fixing the OpenSim/Python handoff and external-load path issues,
  the file-backed selective-ignore model reached IPOPT cleanly. However, the
  solver still followed essentially the same path as the all-compliant branch:
  early `inf_pr ~ 0.147–0.148`, then later drift into `~0.16–0.17`.
- Interpretation: selective rigid tendons for that first subset are technically
  viable, but do not materially break the plateau.

### 20. Tendon slack length review

- Change: compared the local pipeline TSL table
  (`rat-hindlimb-model/data/parameters/tsl_comparison.csv`) with the rerun TSL
  optimization output (`tsl-optimization/new_tsl_values.json`).
- Result: both sources agree on the strongest zero/near-zero TSL muscles:
  `AL`, `CF`, `GMa`, `IP`, `STp`, with `TFL` as the leading near-zero extension.
- Interpretation: the TSL pipeline itself contains real red flags. The first
  selective-ignore subset that was tested did not fully align with the TSL
  evidence because it omitted `CF` and `TFL` and included `GS`.

### 21. Passive fiber force review

- Change: reviewed local code and external Moco/OpenSim guidance on
  `ModOpIgnorePassiveFiberForcesDGF()`.
- Result: official Moco examples and the local `osimpy` wrappers consistently
  ignore passive fiber forces and widen the active force-length curve for
  inverse/tracking robustness.
- Interpretation: re-enabling passive fiber forces is not the most defensible
  next move for this problem and would more likely make the current compliant-
  tendon inverse solve harder before the TSL issues are addressed.

### 22. TSL-aligned selective ignore-tendon-compliance trial

- Change: tested a tendon-slack-length-aligned rigid-tendon subset based on the
  strongest local TSL red flags: bilateral `AL`, `CF`, `GMa`, `IP`, `STp`, and
  `TFL`.
- Result: the solver again reached IPOPT cleanly, but the trajectory was still
  essentially the same as the all-compliant branch: early `inf_pr ~ 0.147–0.148`
  followed by later drift back into the `~0.16–0.17` range.
- Interpretation: even the TSL-aligned subset does not materially break the
  plateau, so the bottleneck is unlikely to be explained mainly by only those
  few worst-TSL muscles.

### 23. First compliant-tendon MocoTrack v2 attempt

- Change: created `/home/hudson/Downloads/CMC_Runs/run_mocotrack_compliant_v2.py`
  as a minimal compliant-tendon fork of the working rigid-tendon v2 tracking
  setup, preserving the v2 mesh and weights while enabling implicit tendon
  compliance.
- Result: the script built and reached IPOPT, but startup feasibility was
  catastrophic: approximately `inf_pr ~ 3.70e+04` at iteration 0 and still on
  that scale at the next iterations.
- Interpretation: this strongly suggested a guess/reference mismatch for the
  compliant tracking problem rather than evidence that the soft-tracking
  direction itself was invalid.

### 24. Real all-compliant inverse warm-start generation

- Change: created `/home/hudson/Downloads/CMC_Runs/run_mocoinverse_compliant_all.py`
  to generate a true all-compliant inverse trajectory on a 20 ms mesh using the
  rigid inverse warm start plus SO-informed tendon-force initialization.
- Result: the run entered IPOPT with good startup feasibility (`iter 0: inf_pr
  ~ 0.443`), reached the familiar restoration neighborhood (`iter 10r: inf_pr
  ~ 0.148`), and then wrote
  `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/MocoInverse_compliant_all/moco_inverse_solution_compliant_all.sto`
  in `1280.8 s` with objective `49.914200`. The log reported
  `Infeasible_Problem_Detected`, but Moco still returned and wrote the
  trajectory.
- Interpretation: a real compliant inverse trajectory can be generated locally,
  and it is a much better-matched warm-start object for compliant tracking than
  patching a rigid-tendon tracking guess.

### 25. Verified compliant inverse trajectory contents

- Change: inspected the written all-compliant inverse output.
- Result: the `.sto` file contains compliant state columns such as
  `/forceset/.../normalized_tendon_force`, confirming it is a true compliant
  trajectory rather than another rigid-tendon inverse output. Its header records
  `num_states=142`, `objective=49.914200`, `solver_duration=1273.738940`, and
  `status=Infeasible_Problem_Detected`.
- Interpretation: the prerequisite for a proper compliant MocoTrack warm start
  is now satisfied.

### 26. Current compliant-tendon MocoTrack v2 rerun status

- Change: relaunched `run_mocotrack_compliant_v2.py` under the
  `opensim-analysis` conda environment after the compliant inverse trajectory
  became available.
- Result: the worker process is actively consuming heavy CPU and the output
  directory contains a new stop-sentinel file. Readable IPOPT output is present
  in `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/mocotrack_compliant_v2.log`,
  but startup feasibility is still catastrophic: `iter 0: inf_pr ~ 3.70e+04`,
  `iter 1: inf_pr ~ 3.70e+04`, `iter 2: inf_pr ~ 3.69e+04`. No tracking
  solution file has been written yet.
- Interpretation: the current rerun is no longer opaque; it is observable and
  still begins on the same bad feasibility scale as the earlier compliant-track
  attempt. That means the real compliant inverse warm start has not yet fixed
  the compliant-tracking startup mismatch by itself.

### 27. DGF fiber damping sweep

- Change: tested runtime `DeGrooteFregly2016Muscle` fiber damping values in the
  exact inverse workflow while keeping the other major solver-side changes fixed.
- Results:
  - `fiber_damping = 0.0` failed badly, with best `inf_pr ~ 0.515` and late
    plateau near `~0.701`.
  - `fiber_damping = 0.1` returned to the familiar bad compliant-inverse regime
    around `~0.23`.
  - `fiber_damping = 0.02` improved substantially (`best inf_pr ~ 0.032`) but
    still ended with `Infeasible_Problem_Detected`.
  - `fiber_damping = 0.05` improved over the old plateau but still ended with
    `Maximum_Iterations_Exceeded`.
  - `fiber_damping = 0.01` was qualitatively different: it reached
    `best inf_pr ~ 0.017`, stayed in that low-feasibility neighborhood for the
    remainder of the run, and wrote a usable `.sto` trajectory instead of
    rebounding into the old `~0.23–0.25` band.
- Interpretation: runtime DGF fiber damping is the first muscle-model lever that
  clearly breaks the old late rebound pattern. The solve still does not reach
  formal IPOPT success, but `0.01` is the strongest exact-inverse branch found
  so far.

### 28. Dedicated rerun of the best damping branch

- Change: reran the exact inverse with `fiber_damping = 0.01`, a dedicated
  output path, a warm start from the previously written low-`inf_pr` solution,
  and a raised iteration budget (`10000`).
- Result: the rerun wrote
  `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/MocoInverse/moco_inverse_solution_damp001_rerun.sto`
  with:
  - `num_iterations=2843`
  - `objective=124.596173`
  - `solver_duration=10362.748899`
  - `status=Restoration_Failed`
  - `success=false`
  During the run, IPOPT again entered the low-feasibility regime and stayed
  there (`inf_pr ~ 0.0176`) instead of rebounding to the old `~0.23–0.25`
  plateau.
- Interpretation: the damping improvement is reproducible. However, the exact
  compliant inverse still ends in restoration failure rather than formal solver
  success, so another DGF-side solve-time lever is still needed.

### 29. GS-specific rigid-tendon follow-up

- Change: used the new targeted override hook to force only `L_GS` and `R_GS`
  into the rigid-tendon subset on top of the `fiber_damping = 0.01` exact
  inverse branch, while warm-starting from the best damping rerun.
- Result: this produced the first exact `MocoInverse` success for the compliant
  workflow. The run wrote
  `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/MocoInverse/moco_inverse_solution_trial_gs_rigid.sto`
  with:
  - `num_iterations=159`
  - `objective=1.349842`
  - `solver_duration=521.284327`
  - `status=Solve_Succeeded`
  - `success=true`
  The solved problem dropped from `140` to `138` states and from `64` to `62`
  derivatives, exactly matching the removal of one implicit
  `normalized_tendon_force` state plus one implicit tendon-force derivative for
  each of the two GS muscles.
- Interpretation: bilateral GS tendon compliance was a critical blocker in the
  exact prescribed-motion inverse. This is the first branch that moved the
  problem from “better failed restore” to a formal successful solve.

### 30. GS-rigid plus targeted reserve overrides

- Change: kept the successful GS-rigid exact-inverse formulation and added named
  reserve optimal-force overrides for the dominant reserve-heavy coordinates:
  `sacrum_y:3.0`, `sacrum_z:0.5`, `hip_r_flx:0.5`, `hip_l_flx:0.5`, warm-started
  from the successful GS-rigid `.sto`.
- Result: the run wrote
  `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/MocoInverse/moco_inverse_solution_trial_gs_rigid_coordinate_reserves.sto`
  with:
  - `num_iterations=366`
  - `objective=1.211161`
  - `solver_duration=1191.200233`
  - `status=Solve_Succeeded`
  - `success=true`
  Relative to the first GS-rigid success, the solution improved the objective
  and substantially reduced the main reserve outliers:
  - `sacrum_y max|tau|: 1.7822 -> 0.8911`
  - `sacrum_z max|tau|: 0.5822 -> 0.1164`
- Interpretation: once GS tendon compliance was removed, the remaining dominant
  issue was not convergence itself but reserve plausibility. Coordinate-specific
  reserve overrides can improve that while preserving exact-inverse success.

### 31. GS-rigid plus reserve overrides plus sacrum_x follow-up

- Change: added `sacrum_x:0.5` on top of the already successful GS-rigid plus
  reserve-override branch and warm-started from that branch’s successful `.sto`.
- Result: the run wrote
  `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/MocoInverse/moco_inverse_solution_trial_gs_rigid_coordinate_reserves_sacrumx.sto`
  with:
  - `num_iterations=260`
  - `objective=1.189695`
  - `solver_duration=1162.166700`
  - `status=Solve_Succeeded`
  - `success=true`
  This further improved the best exact-inverse objective and materially reduced
  the remaining largest sacrum reserve outlier:
  - `sacrum_x max|tau|: 0.3916 -> 0.0783`
  while preserving the earlier `sacrum_y` and `sacrum_z` gains.
- Interpretation: this is now the best exact `MocoInverse` branch found so far.
  The problem is no longer “can the compliant exact inverse converge?” but
  “which localized rigid-tendon and reserve settings give the best physiological
  exact-inverse solution without losing convergence?”

## Current Best Configurations

### Best exact-inverse compliant branch explored so far

- mesh interval: `0.01`
- convergence tolerance: `1e-3`
- constraint tolerance: `1e-4`
- rigid-tendon MocoInverse warm start
- SO-informed initialization of `normalized_tendon_force`
- no initial velocity equilibrium goal
- `minimize_implicit_auxiliary_derivatives = True`
- `implicit_auxiliary_derivatives_weight` anywhere in `0.0–0.01` behaves
  similarly

This older branch enters IPOPT reliably, reaches roughly `inf_pr ~ 0.146–0.148`
early, and then stalls in the `~0.16–0.17` range instead of converging.

### Current best exact-inverse branch overall

- runtime DGF fiber damping: `0.01`
- thresholded rigid-tendon subset from `Walk TSL <= 0.5 mm`
- extra rigid-tendon overrides: `L_GS`, `R_GS`
- coordinate reserve overrides:
  - `sacrum_x:0.5`
  - `sacrum_y:3.0`
  - `sacrum_z:0.5`
  - `hip_r_flx:0.5`
  - `hip_l_flx:0.5`
- warm start:
  `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/MocoInverse/moco_inverse_solution_trial_gs_rigid_coordinate_reserves.sto`
- current best output:
  `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/MocoInverse/moco_inverse_solution_trial_gs_rigid_coordinate_reserves_sacrumx.sto`
- current best status: `Solve_Succeeded`
- current best objective: `1.189695`

This is now a formal exact-inverse success, not just a better failed restore.
The decisive structural change was rigid tendon for bilateral GS, and the next
improvements came from coordinate-specific reserve overrides that reduced the
dominant sacrum reserve loads while preserving convergence.

### Best current warm-start object for compliant tracking

- file:
  `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/MocoInverse_compliant_all/moco_inverse_solution_compliant_all.sto`
- source script:
  `/home/hudson/Downloads/CMC_Runs/run_mocoinverse_compliant_all.py`
- properties:
  - all-compliant inverse trajectory
  - includes `normalized_tendon_force` states
  - generated from the same Walk05 problem family now being used for compliant
    tracking

This is now the most defensible initial guess source for compliant MocoTrack v2.

## Practical Conclusion

The working interpretation has changed materially.

The exact prescribed-motion compliant-tendon inverse problem **can** be solved
successfully in this workflow. The broad late-stage restoration failure was not
the final story. Two things turned out to matter decisively:

- runtime DGF fiber damping (`0.01`) moved the solve into a much better
  feasibility regime,
- bilateral GS rigid tendon removed the remaining local tendon-compliance
  blocker and produced the first exact-inverse success.

Once that success existed, the remaining problem changed from convergence to
solution quality. Targeted coordinate-specific reserve overrides then improved
the exact-inverse branch further by reducing the dominant sacrum reserve loads
while preserving solver success.

The current best branch therefore supports this narrower interpretation:

- GS tendon compliance was a critical local blocker,
- sacrum reserve usage was a secondary plausibility/quality issue,
- broad source-anatomy retuning was not needed to reach exact-inverse success.

## Static Optimization as an Initial Guess

Static Optimization results are available locally:

- `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/scaled_scaled_StaticOptimization_activation.sto`
- `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/scaled_scaled_StaticOptimization_force.sto`
- `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/SO_physiology/scaled_scaled_StaticOptimization_activation.sto`
- `/home/hudson/Downloads/CMC_Runs/BAA01/Baseline/SO_physiology/scaled_scaled_StaticOptimization_force.sto`

However, the existing rigid-tendon MocoInverse solution is still the stronger
initial guess source for the compliant-tendon MocoInverse problem because it
already matches the Moco state/control layout much more closely. Static
Optimization was still useful as a source of muscle-force information for the
improved SO-informed tendon-force guess.

## Additional documents created

- `/home/hudson/Downloads/CMC_Runs/tendon_slack_length_review.md`
  - ranked review of zero/near-zero TSL muscles from the local pipeline table
    and rerun TSL optimization output

## Current next step

The current next step is no longer another global convergence rescue. The best
exact-inverse branch is now successful. The next work should focus on
interpreting and improving the physiological plausibility of the successful
solution, starting with the few muscles that now appear closest to saturation
(for example `R_FDL` in the current best branch) before reopening broader model
changes.
