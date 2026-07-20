#!/bin/bash
#SBATCH --job-name=TriGlobal_Dev
#SBATCH --output=triglobal_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==========================================="
echo " TRI-GLOBAL HPC SLURM JOB"
echo "==========================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Nodes       : $SLURM_JOB_NUM_NODES"
echo " MPI ranks   : $SLURM_NTASKS"
echo " Project root: $PROJECT_ROOT"
echo "==========================================="

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tri_engine

cd "$PROJECT_ROOT"
mpiexec -n $SLURM_NTASKS python main.py
