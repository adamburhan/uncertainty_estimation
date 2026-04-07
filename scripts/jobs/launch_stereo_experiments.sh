#!/bin/bash
# Launch the four stereo-covariance ablation experiments described in
# scripts/jobs/README_stereo_experiments.md, using Hydra multirun + submitit.
#
# Usage:
#   ./scripts/jobs/launch_stereo_experiments.sh A         # only Experiment A
#   ./scripts/jobs/launch_stereo_experiments.sh A B C D   # all four
#   ./scripts/jobs/launch_stereo_experiments.sh all
#
# Each experiment is one Hydra multirun = one SLURM array. Cells overlap on
# purpose: B/C/D add only the *new* cells beyond A so total compute is minimal.
# Group names already collide for shared cells, so wandb will aggregate them
# into the same averaged curve regardless of which experiment launched them.
#
# IMPORTANT: do NOT re-launch a stereo cell that already ran under another
# experiment label. The whole point of the overlap-and-skip design is that
# every cell trains exactly once.

set -euo pipefail

REPO_DIR=/home/mila/a/adam.burhan/repos/uncertainty_estimation
cd "$REPO_DIR"

# All 10 stereo configs (Experiment A spans them all)
ALL_STEREO="horizontal_5cm,horizontal_10cm,horizontal_20cm,horizontal_50cm,horizontal_100cm,vertical_5cm,vertical_10cm,vertical_20cm,vertical_50cm,vertical_100cm"
# 4-config subset for B/C/D — extreme baselines per direction, max signal
SUBSET_STEREO="horizontal_5cm,horizontal_100cm,vertical_5cm,vertical_100cm"
SEEDS="42,0,2026"

# Default sigma values — flag for sanity check before launching
SIGMA_2D=2.0      # pixels
SIGMA_3D=0.05     # metres (5cm; tune relative to scene depth scale)

LAUNCHER_OVERRIDES="--multirun hydra/launcher=sbatch +hydra/sweep=sbatch"

run_A() {
  echo ">>> Experiment A: stereo effect, 10 configs x 3 seeds = 30 runs"
  uv run -m scripts.train_stereo_cov $LAUNCHER_OVERRIDES \
    experiment.label=A_stereo \
    loss.name=bearing_nll \
    correspondence.mode=real \
    dataset.stereo_config=$ALL_STEREO \
    training.seed=$SEEDS \
    dataset=sss
}

run_B() {
  echo ">>> Experiment B: matcher-falsification, 4 configs x synthetic_2d x 3 seeds = 12 runs"
  uv run python scripts/train_stereo_cov.py $LAUNCHER_OVERRIDES \
    experiment.label=B_falsif \
    loss.name=bearing_nll \
    correspondence.mode=synthetic \
    correspondence.sigma=$SIGMA_2D \
    dataset.stereo_config=$SUBSET_STEREO \
    training.seed=$SEEDS \
    dataset=sss
}

run_C() {
  echo ">>> Experiment C: loss ablation, 4 configs x pixel_nll x 3 seeds = 12 runs"
  uv run python scripts/train_stereo_cov.py $LAUNCHER_OVERRIDES \
    experiment.label=C_loss \
    loss.name=pixel_nll \
    correspondence.mode=real \
    dataset.stereo_config=$SUBSET_STEREO \
    training.seed=$SEEDS \
    dataset=sss
}

run_D() {
  echo ">>> Experiment D: 3D positive control, 4 configs x synthetic_3d x 3 seeds = 12 runs"
  uv run python scripts/train_stereo_cov.py $LAUNCHER_OVERRIDES \
    experiment.label=D_3dctrl \
    loss.name=bearing_nll \
    correspondence.mode=synthetic_3d \
    correspondence.sigma=$SIGMA_3D \
    dataset.stereo_config=$SUBSET_STEREO \
    training.seed=$SEEDS \
    dataset=sss
}

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 [A|B|C|D|all] ..."
  exit 1
fi

for arg in "$@"; do
  case "$arg" in
    A)   run_A ;;
    B)   run_B ;;
    C)   run_C ;;
    D)   run_D ;;
    all) run_A; run_B; run_C; run_D ;;
    *)   echo "Unknown experiment: $arg" >&2; exit 1 ;;
  esac
done
