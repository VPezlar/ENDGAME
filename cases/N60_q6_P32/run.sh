#!/bin/bash
#PBS -N ENDGAME_N60_q6_P32
#PBS -q zeus_all_q
#PBS -l nodes=2:ppn=16
#PBS -l walltime=06:00:00
#PBS -j oe

# Redirect output immediately to a log file on NFS.
# This bypasses PBS output buffering so tail -f works in real time.
mkdir -p "$PBS_O_WORKDIR/logs"
exec > "$PBS_O_WORKDIR/logs/run.log" 2>&1

# PBS_O_WORKDIR = this case directory (qsub submitted from here)
NPROCS=$(wc -l < $PBS_NODEFILE)
PYTHON="$HOME/miniconda3/envs/tri_engine_complex/bin/python"

echo "ENDGAME case: N60_q6_P32  Job: $PBS_JOBID  Ranks: $NPROCS"

export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ENDGAME_NX=60 ENDGAME_Q=6
export ENDGAME_MODES=100 ENDGAME_NCV=300 ENDGAME_TARGET=43.0

$(dirname "$PYTHON")/mpiexec -n $NPROCS --bind-to none --hostfile $PBS_NODEFILE \
    -x PYTHONDONTWRITEBYTECODE \
    -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS \
    -x ENDGAME_NX -x ENDGAME_Q \
    -x ENDGAME_MODES -x ENDGAME_NCV -x ENDGAME_TARGET \
    "$PYTHON" -u "$PBS_O_WORKDIR/main.py"
