#!/bin/bash


for loss in bearing_nll pixel_nll; do
  for arch in UNetXS UNetSmall UNetM; do
    sbatch scripts/jobs/slurm_job.sh \
      experiment.name="001_${loss}_${arch}" \
      loss.name=$loss \
      model.architecture=$arch \
      training.device=cuda
  done
done
