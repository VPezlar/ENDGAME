#!/bin/bash
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

# Disable .pyc caching — prevents stale NFS bytecode from being used
export PYTHONDONTWRITEBYTECODE=1

# Purge any cached bytecode files
find "$PBS_O_WORKDIR/src" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find "$PBS_O_WORKDIR/src" -name '*.pyc' -delete 2>/dev/null || true

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export ENDGAME_NX ENDGAME_Q ENDGAME_MODES ENDGAME_NCV ENDGAME_TARGET

# Pass PYTHONDONTWRITEBYTECODE explicitly to every remote MPI rank via -x.
# Without -x, OpenMPI may not propagate all shell env vars to remote nodes.
mpiexec -n $NPROCS \
    -x PYTHONDONTWRITEBYTECODE \
    -x OMP_NUM_THREADS \
    -x OPENBLAS_NUM_THREADS \
    -x ENDGAME_NX \
    -x ENDGAME_Q \
    -x ENDGAME_MODES \
    -x ENDGAME_NCV \
    -x ENDGAME_TARGET \
    "$PYTHON" -u main.py
