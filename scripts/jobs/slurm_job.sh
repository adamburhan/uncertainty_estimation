#!/bin/bash
# Usage:
#   sbatch scripts/jobs/slurm_job.sh experiment.name=001_bearing_gt [extra_hydra_overrides...]
#
# The first positional arg MUST be experiment.name=<name> so the job and
# WandB run are named correctly. All remaining args are forwarded to Hydra.
#
# Examples:
#   sbatch scripts/jobs/slurm_job.sh experiment.name=001_bearing_gt
#   sbatch scripts/jobs/slurm_job.sh experiment.name=002_pixel_nll loss.name=pixel_nll
#   sbatch --config-name base_kitti scripts/jobs/slurm_job.sh experiment.name=003_kitti

#SBATCH --output=logs/slurm-%j.out
#SBATCH --account=mila
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=50:00:00

set -euo pipefail

REPO_DIR=/home/mila/a/adam.burhan/repos/uncertainty_estimation
cd "$REPO_DIR"

mkdir -p logs

echo "=== Job info ==="
echo "Date:      $(date)"
echo "Node:      $(hostname)"
echo "Job ID:    $SLURM_JOB_ID"
echo "Git:       $(git rev-parse --short HEAD)"
echo "Args:      $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================"

# data staging
root=/home/mila/a/adam.burhan/scratch/datasets/tartanair-v2

mkdir -p /tmp/tartanair-v2

cp -r "$root/ArchVizTinyHouseDay" /tmp/tartanair-v2/

cd /tmp/tartanair-v2/ArchVizTinyHouseDay
for zip in *.zip; do
    unzip -q -n "$zip"
done

cd "$REPO_DIR"
uv run python scripts/train_stereo_cov.py "$@"
