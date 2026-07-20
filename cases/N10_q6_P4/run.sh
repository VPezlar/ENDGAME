#!/bin/bash
#PBS -N ENDGAME_N10_q6_P4
#PBS -q zeus_short_q
#PBS -l nodes=1:ppn=16
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o logs/run.log

# PBS_O_WORKDIR = this case directory (qsub submitted from here)
NPROCS=$(wc -l < $PBS_NODEFILE)
PYTHON="$HOME/miniconda3/envs/tri_engine/bin/python"

echo "ENDGAME case: N10_q6_P4  Job: $PBS_JOBID  Ranks: $NPROCS"

export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ENDGAME_NX=10 ENDGAME_Q=6
export ENDGAME_MODES=100 ENDGAME_NCV=300 ENDGAME_TARGET=43.0

mpiexec -n $NPROCS \
    -x PYTHONDONTWRITEBYTECODE \
    -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS \
    -x ENDGAME_NX -x ENDGAME_Q \
    -x ENDGAME_MODES -x ENDGAME_NCV -x ENDGAME_TARGET \
    "$PYTHON" -u "$PBS_O_WORKDIR/main.py"
