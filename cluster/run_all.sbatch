#!/bin/bash
# The six analyses in the paper (landmarks 2, 6 and 12 h x splits A and B), one
# array task each, on a SLURM cluster with GPUs (TabPFN needs one; everything
# else runs on the CPUs). The published analyses used 28 CPUs + 1 GPU per task.
#
#   sbatch cluster/run_all.sbatch           # from the repository root
#
# Edit the module and environment lines for your cluster.
#SBATCH --job-name=hfnc-rox
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail

LANDMARKS=(2 2 6 6 12 12)
SPLITS=(A B A B A B)
L=${LANDMARKS[$SLURM_ARRAY_TASK_ID]}
S=${SPLITS[$SLURM_ARRAY_TASK_ID]}

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4 CUDA/12.6.0    # Warwick Avon
source "${HOME}/hfnc_env/bin/activate"

echo "start $(date -Is) | job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} | landmark ${L} h split ${S}"
python -u scripts/2_run_analysis.py --landmark "$L" --split "$S" --jobs "$SLURM_CPUS_PER_TASK"
echo "end $(date -Is)"
