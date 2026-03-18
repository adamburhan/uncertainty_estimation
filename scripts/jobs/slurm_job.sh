#!/bin/bash
#SBATCH --job-name=stereo_ssl
#SBATCH --output=logs/stereo_ssl_%j.out
#SBATCH --error=logs/stereo_ssl_%j.err
#SBATCH --account=rrg-lpaull
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_DIR=/home/mila/a/adam.burhan/repos/uncertainty_estimation
cd "$REPO_DIR"

# Make sure log dir exists
mkdir -p logs

# Diagnostic info
echo "=== Job info ==="
echo "Date:      $(date)"
echo "Node:      $(hostname)"
echo "Job ID:    $SLURM_JOB_ID"
echo "Git:       $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================"

uv run python scripts/train_stereo_cov.py --config configs/training/stereo_ssl.yaml
