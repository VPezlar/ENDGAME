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

# ---- Isolated run directory ----
# $TMPDIR = /gtmp/pbs.{JOBID}/ set by PBS, unique per job.
# Rsyncing the project here gives each job its own private copy of
# FDq/inputs/, FDq/output/, and src/. No two jobs touch the same files.
RUNDIR="$TMPDIR/endgame_run"
mkdir -p "$RUNDIR"
rsync -a \
    --exclude='.git' --exclude='output' --exclude='logs' \
    --exclude='runs'  --exclude='*.pyc' --exclude='__pycache__' \
    "$PBS_O_WORKDIR/" "$RUNDIR/"
mkdir -p "$RUNDIR/output"

echo "========================================"
echo " ENDGAME scaling job"
echo "========================================"
echo " Job ID   : $PBS_JOBID"
echo " Nodes    : $(sort -u $PBS_NODEFILE | tr '\n' ' ')"
echo " Ranks    : $NPROCS"
echo " N        : $ENDGAME_NX"
echo " RUNDIR   : $RUNDIR"
echo "========================================"

export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ENDGAME_NX ENDGAME_Q ENDGAME_MODES ENDGAME_NCV ENDGAME_TARGET

# Run main.py by ABSOLUTE PATH.
# main.py immediately does os.chdir(dirname(__file__)) so every rank,
# regardless of where mpiexec spawned it, works from RUNDIR.
mpiexec -n $NPROCS \
    -x PYTHONDONTWRITEBYTECODE \
    -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS \
    -x ENDGAME_NX -x ENDGAME_Q \
    -x ENDGAME_MODES -x ENDGAME_NCV -x ENDGAME_TARGET \
    "$PYTHON" -u "$RUNDIR/main.py"

# Copy results back to permanent project directory
rsync -a "$RUNDIR/output/" "$PBS_O_WORKDIR/output/"
