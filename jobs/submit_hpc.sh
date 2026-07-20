#!/bin/bash
#PBS -N ENDGAME
#PBS -q zeus_all_q
#PBS -l nodes=1:ppn=16
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o logs/endgame_pbs.log

cd $PBS_O_WORKDIR
mkdir -p logs

# PBS Pro does not set $PBS_NP; count from the nodefile instead
NPROCS=$(wc -l < $PBS_NODEFILE)

echo "==========================================="
echo " ENDGAME — PBS JOB"
echo "==========================================="
echo " Job ID      : $PBS_JOBID"
echo " Node list   : $(sort -u $PBS_NODEFILE | tr '\n' ' ')"
echo " MPI ranks   : $NPROCS"
echo " Working dir : $PBS_O_WORKDIR"
echo "==========================================="

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tri_engine

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mpiexec -n $NPROCS python -u main.py
