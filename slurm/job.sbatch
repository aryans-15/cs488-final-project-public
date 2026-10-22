#!/bin/bash
#SBATCH --job-name=hd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
# usage: sbatch [--account=A --partition=P] slurm/job.sbatch ENV MODULE [args]
#   e.g. sbatch slurm/job.sbatch train hackdistill.train grpo --tasks "$TASKS" --model "$MODEL" --out-dir OUT --variant loophole --seed 1 --steps 40
source "$HD_CODE/slurm/env.sh" "$1"
cd "$HD_CODE"
module=$2; shift 2
python -m "$module" "$@"
