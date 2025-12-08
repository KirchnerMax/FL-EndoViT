#!/bin/bash
#
# Job-array submitter: enqueues one GPU job per array task. 12 tasks in total
# (4 folds × 3 models). Each array task consumes 1 GPU and writes logs to
# logs/<jobname>_<jobid>_<arrayid>.out/.err. Create the logs/ directory
# before submitting: mkdir -p logs
#
# Usage: sbatch start.sh
#
# Resources and limits: adjust --time, --cpus-per-task, --mem as needed
#
#SBATCH --job-name=gynsurg_bleeding_finetune
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# Commands to run (one per array index). Keep the same working dir as submission.
COMMANDS=(
	"python main_MAE_CEN.py --kfold 0"
	"python main_MAE_CEN.py --kfold 1"
	"python main_MAE_CEN.py --kfold 2"
	"python main_MAE_CEN.py --kfold 3"
	"python main_MAE_FL.py --kfold 0"
	"python main_MAE_FL.py --kfold 1"
	"python main_MAE_FL.py --kfold 2"
	"python main_MAE_FL.py --kfold 3"
	"python main_ResNet.py --kfold 0"
	"python main_ResNet.py --kfold 1"
	"python main_ResNet.py --kfold 2"
	"python main_ResNet.py --kfold 3"
)

# Sanity: ensure array index in range
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -lt 0 ] || [ "$IDX" -ge "${#COMMANDS[@]}" ]; then
	echo "Invalid SLURM_ARRAY_TASK_ID=$IDX" >&2
	exit 2
fi

CMD=${COMMANDS[$IDX]}

echo "Running task $IDX: $CMD"

# create logs dir just in case (best if created on submission host beforehand)
mkdir -p logs

# run the selected command using srun so Slurm binds a single GPU to the process
# use SLURM_CPUS_PER_TASK if set, otherwise default to 4
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-4}
echo "Launching with srun: GPUs=1 CPUs=${CPUS_PER_TASK}"
# --gres is redundant because the job requests 1 GPU already, but specify for clarity
srun --gres=gpu:1 --cpus-per-task=${CPUS_PER_TASK} --gpu-bind=single:1 $CMD


