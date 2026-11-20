# Iterative AIMMD-TIS on the Wolfe–Quapp potential

A complete, runnable AIMMD-TIS loop on a 2-D toy potential with an exact reference committor. Use it to see how the method works and which diagnostics tell you whether it is working.

**Start with `aimmd_tis_walkthrough.ipynb`.** It goes through a finished run iteration by iteration.

## The loop (`run_iterative.py`)
1. **Seed:** short AIMMD-TPS gives the first logit committor q₀. Stable-state MD in A and B gives the flux and each state's q range.
2. **Interfaces** are placed on q_k between max q in A and min q in B.
3. **TIS** runs forward and backward, one process per interface. Path descriptors, shot results and path extrema are recorded while sampling (descriptor storages and extrema CSVs), so the OPS storages are never read back.
4. **Reweighting:** MultiSet-MBAR (`rpe_mbar`) combines all interface sets q₀…q_k into the reweighted path ensemble RPE_k. Stable-state MD is added by flux matching at the joined first interface.
5. **Retraining:** q_{k+1} is trained on RPE_k (weighted log-likelihood plus smoothness), then the loop repeats until the model change Δq stays below `run.convergence.delta_q_tol`.

Every stage writes a `.done` marker and is skipped on rerun, so an interrupted run resumes with the same command.

```bash
export PYTHONPATH=~/code/rpe_mbar/src          # unless rpe_mbar is pip-installed
python run_iterative.py --quick --out quick_results                  # smoke test, minutes
python run_iterative.py --out results_it_100_steps --tis-steps 100 --until-converged
```

## Diagnostics (`diagnostics.py`, shown in the walkthrough)
| Figure | Question it answers | Good result |
|---|---|---|
| Interfaces on q_k | where does TIS sample? | interfaces bend around the states and follow the channels |
| Crossing probability vs ideal | is the model globally a committor? | RPE curves on (1+e^−q)/(1+e^−q_i); no dips or jumps ("bubbles") |
| Chevron plot | is training balanced along q? | I(q) on the ideal curve; flat loss per point; p_A, p_B on the logistic curves |
| Calibration | does the model q match the measured committor? | points on the diagonal over the full q range |
| Landscapes | what do RPE, data, model and theory look like? | model projection ≈ shot committor ≈ exact; small error maps |
| ρ(ξ\|q), unit gradients | which descriptor drives progress, and where? | interpretable, smooth trends |
| Convergence | is the loop done? | rates and calibration settle; Δq below tolerance |

The diagnostics follow Breebaart & Bolhuis, J. Chem. Phys. 165, 124125 (2026).

## Files
```
run_iterative.py              the loop (CLI; stages: tps, stable, iterate)
diagnostics.py                all diagnostics and plots; per-iteration figures in <run>/analysis/it{k}/
aimmd_tis_walkthrough.ipynb   the guided analysis of a run
monitor_run.ipynb, monitor.py live view of a running job
config.json                   system, model, training, run and analysis settings ('quick' block for --quick)
theory/                       exact committor for the configured potential (computed once)
extras/convergence.py         bootstrap errors of crossing probabilities and rates over iterations
extras/retrain_control.py     offline training studies (scratch vs warm start, learning rate, smoothness)
results/, results_it_100_steps/   example runs (300 and 100 TIS steps per interface)
```
