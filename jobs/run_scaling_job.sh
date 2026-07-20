#!/bin/bash
#PBS -N ENDGAME_scaling
#PBS -q zeus_all_q
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o logs/scaling_pbs.log

mkdir -p "$PBS_O_WORKDIR/logs"

NPROCS=$(wc -l < $PBS_NODEFILE)

: ${ENDGAME_NX:=30}
: ${ENDGAME_Q:=6}
: ${ENDGAME_MODES:=100}
: ${ENDGAME_NCV:=300}
: ${ENDGAME_TARGET:=43.0}

PYTHON="$HOME/miniconda3/envs/tri_engine/bin/python"

# -------------------------------------------------------
# Create an isolated per-job working directory.
# $TMPDIR is set by PBS to /gtmp/pbs.{JOBID}/ — unique
# per job. Rsyncing the project here means:
#   - Each job has its own FDq/inputs/ and FDq/output/
#   - Each job has its own src/ (no shared __pycache__)
#   - Concurrent jobs cannot interfere at all
# -------------------------------------------------------
RUNDIR="$TMPDIR/endgame_run"
mkdir -p "$RUNDIR"

rsync -a \
    --exclude='.git' \
    --exclude='output' \
    --exclude='logs' \
    --exclude='runs' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    "$PBS_O_WORKDIR/" "$RUNDIR/"

mkdir -p "$RUNDIR/output" "$RUNDIR/logs"

cd "$RUNDIR"

echo "========================================"
echo " ENDGAME scaling job"
echo "========================================"
echo " Job ID   : $PBS_JOBID"
echo " Nodes    : $(sort -u $PBS_NODEFILE | tr '\n' ' ')"
echo " Ranks    : $NPROCS"
echo " Python   : $PYTHON"
echo " N        : $ENDGAME_NX"
echo " RUNDIR   : $RUNDIR"
echo " Out tag  : N${ENDGAME_NX}_q${ENDGAME_Q}_P${NPROCS}"
echo "========================================"

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export ENDGAME_NX ENDGAME_Q ENDGAME_MODES ENDGAME_NCV ENDGAME_TARGET

mpiexec -n $NPROCS \
    -x PYTHONDONTWRITEBYTECODE \
    -x OMP_NUM_THREADS \
    -x OPENBLAS_NUM_THREADS \
    -x ENDGAME_NX -x ENDGAME_Q \
    -x ENDGAME_MODES -x ENDGAME_NCV -x ENDGAME_TARGET \
    "$PYTHON" -u main.py

# Copy results back to permanent project directory
rsync -a "$RUNDIR/output/" "$PBS_O_WORKDIR/output/"
