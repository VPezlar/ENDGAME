#!/bin/bash
# Generic PBS template for the ENDGAME scaling study.
# Parameters passed via: qsub -v "ENDGAME_NX=60,..." run_scaling_job.sh

#PBS -N ENDGAME_scaling
#PBS -q zeus_all_q
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o logs/scaling_pbs.log

cd $PBS_O_WORKDIR
mkdir -p logs

NPROCS=$(wc -l < $PBS_NODEFILE)

: ${ENDGAME_NX:=30}
: ${ENDGAME_Q:=6}
: ${ENDGAME_MODES:=100}
: ${ENDGAME_NCV:=300}
: ${ENDGAME_TARGET:=43.0}

PYTHON="$HOME/miniconda3/envs/tri_engine/bin/python"

echo "========================================"
echo " ENDGAME scaling job"
echo "========================================"
echo " Job ID   : $PBS_JOBID"
echo " Nodes    : $(sort -u $PBS_NODEFILE | tr '\n' ' ')"
echo " Ranks    : $NPROCS"
echo " Python   : $PYTHON"
echo " N        : $ENDGAME_NX"
echo " Modes/NCV: $ENDGAME_MODES / $ENDGAME_NCV"
echo " Out tag  : N${ENDGAME_NX}_q${ENDGAME_Q}_P${NPROCS}"
echo "========================================"

# CRITICAL: disable Python bytecode caching.
# On NFS clusters, compute nodes can cache stale .pyc files and run old code
# even after the .py source has been updated. This silently produces wrong
# results. PYTHONDONTWRITEBYTECODE=1 forces Python to always interpret the
# .py file directly — no .pyc is read or written, ever.
export PYTHONDONTWRITEBYTECODE=1

# Also delete any existing cache just in case
find "$PBS_O_WORKDIR/src" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find "$PBS_O_WORKDIR/src" -name '*.pyc' -delete 2>/dev/null || true

# Prevent OpenBLAS/MUMPS from spawning extra threads per rank
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export ENDGAME_NX ENDGAME_Q ENDGAME_MODES ENDGAME_NCV ENDGAME_TARGET

mpiexec -n $NPROCS "$PYTHON" -u main.py
