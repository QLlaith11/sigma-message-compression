# Post-hoc message compression for learned decentralized MAPF

Code for compressing the messages exchanged by the agents of a trained SIGMA policy (kept frozen) to a few bits per message, and for the evaluation, controls and negative-result experiments around it.

## Layout

```
src/
  train_and_evaluate_all_phases.py     training of missing compressors and evaluation phases A-D (house and maze)
  train_compressors.py                 training of the learned compressors (house / maze, several bit budgets and seeds)
  baselines_and_robustness.py          matched-bit baselines, measured bit rates, message loss / corruption / checksum study
  controls_and_message_probe.py        no-code decoder, shuffled-code control, significance tests, probe of the transmitted symbols
exploratory/
  communication_free_prediction_and_finetuning.py   no-communication fine-tuning and neighbour-state predictors
  sheaf_predictor_supervised.py                      supervised sheaf-diffusion predictor, MLP and mean baselines, DAgger
  oracle_message_diagnostic.py                       oracle and noisy-oracle neighbour tokens
  oracle_goal_bit_and_stillness_flag.py              one-bit at-goal flag and stillness-based flag
  oracle_feature_ablation.py                         hand-designed sender features
  message_probe_first_version.py                     probe of the transmitted symbols
  learned_compressor_first_version.py                first version of the learned compressor
  closed_loop_pca_compression.py                     closed-loop PCA message compression
  open_loop_pca_compression.py                       PCA message compression with teacher-shadow vectors
  communication_ablation_inference_only.py           inference-only ablations of the attention exchange
requirements.txt
```

## Requirements

Python 3.10+, PyTorch, NumPy, pandas, matplotlib, scikit-image, tqdm (`requirements.txt`). A GPU is strongly recommended. The scripts are written as single Colab cells and were run on Google Colab (PyTorch 2.11-2.14, NumPy 2.1-2.4).

## Data and checkpoints

Each script clones https://github.com/marmotlab/SIGMA (or asks for its zip) and uses its released checkpoints and fixed test sets:

- house maps: `models/save_model/model_house/84000_house.pth`, `house_test_set/`
- maze maps: `models/save_model/model_maze/82000.pth`, `test_set/`

The SIGMA files are not part of this repository and keep their own license. Before importing the repository's `environment.py` the scripts apply a one-line compatibility change needed with NumPy >= 2.2 that keeps the original behaviour.

## Running

1. Open a script in a Colab cell with a GPU runtime. `train_and_evaluate_all_phases.py` and `controls_and_message_probe.py` mount Google Drive and keep compressors and results in `MyDrive/sigma_paper` (`USE_DRIVE`); set `OUT_DIR` otherwise.
2. Set `SMOKE = True` for a short test of the whole script, then `SMOKE = False`.
3. `train_and_evaluate_all_phases.py` selects phases with `PHASES`:
   - `A` second training seed on house maps (32 and 64 agents)
   - `B` 128 agents on house maps
   - `C` maze maps: training, rate-versus-performance study with baselines, optional robustness (`MAZE_ROBUST`)
   - `D` 200-case evaluation of the headline configurations (`HEADLINE_CONFIGS`, `D_ENVS`)
   Missing compressors are trained automatically; finished results are reused unless `FORCE = True`.
4. `controls_and_message_probe.py` runs the controls on the trained compressors (the no-code decoder is trained if missing) and probes the transmitted symbols (`CTRL_ENVS`, `CTRL_SEEDS`, `CTRL_PLAN`, `PROBE_RUNS`).
5. `baselines_and_robustness.py` expects compressors in `OUT_DIR/<env>/` as written by `train_compressors.py` or by the phase script (`learned_k{k}_b{bits}_seed{seed}.pt`).

## Outputs

`results_*.csv`, `pareto*.png` and probe tables in `OUT_DIR/<env>/`. Each result row reports success rate with a 95% Wilson interval, makespan, mean per-agent arrival time and measured bits received per agent per step.

## Definitions

- Message: the 256-d agent state before each of the two attention exchanges; at full precision 2 x 256 x 32 = 16,384 bits per message.
- Bits per agent-step: measured messages received per agent per step times bits per message (no entropy coding or headers).
- Success: all agents on their goals within 512 steps.
- Compressor `(k, bits)`: k numbers per exchange at `bits` bits each, so 2 x k x bits bits per message; `(0, 1)` is the decoder without a code.
