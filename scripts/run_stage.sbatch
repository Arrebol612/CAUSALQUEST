#!/usr/bin/env bash
#SBATCH --job-name=cq-framework
#SBATCH --partition=MGPU-TC2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=05:30:00
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
set -euo pipefail
# Activate your environment and mkdir -p outputs before sbatch. Arguments are explicit cq commands.
"${CQ_PYTHON:-python}" -m causalquest "$@"

