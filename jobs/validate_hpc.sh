#!/bin/bash
#PBS -N ENDGAME_val
#PBS -q zeus_short_q
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o logs/endgame_val.log

cd $PBS_O_WORKDIR
mkdir -p logs

# PBS Pro does not set $PBS_NP; count from the nodefile instead
NPROCS=$(wc -l < $PBS_NODEFILE)

echo "=== ENDGAME VALIDATION RUN ==="
echo "Job: $PBS_JOBID  |  Ranks: $NPROCS  |  Dir: $PBS_O_WORKDIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tri_engine

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mpiexec -n $NPROCS python -u validate.py
