#!/bin/bash
# Generic single-run PBS script for the ENDGAME scaling study.
# Parameters are passed via qsub -v, e.g.:
#   qsub -v "ENDGAME_NX=60" -l nodes=4:ppn=16 run_scaling_job.sh

#PBS -N ENDGAME_scaling
#PBS -q zeus_all_q
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o logs/scaling_pbs.log

cd $PBS_O_WORKDIR
mkdir -p logs

# Count total MPI ranks from the nodefile (PBS Pro: $PBS_NP is not set)
NPROCS=$(wc -l < $PBS_NODEFILE)

# Default params — overridden by qsub -v
: ${ENDGAME_NX:=30}
: ${ENDGAME_Q:=6}
: ${ENDGAME_MODES:=100}
: ${ENDGAME_NCV:=300}
: ${ENDGAME_TARGET:=43.0}

# Use the absolute path to Python from the conda env.
# This is the key fix for multi-node jobs: conda activate only sets up
# the shell on the head node. Remote nodes spawned by mpiexec via SSH
# do not inherit the shell environment, so `python` is not found.
# Using the absolute path means every rank — on every node — finds the
# correct interpreter with petsc4py/slepc4py/mpi4py already installed.
PYTHON="$HOME/miniconda3/envs/tri_engine/bin/python"

echo "========================================"
echo " ENDGAME scaling job"
echo "========================================"
echo " Job ID      : $PBS_JOBID"
echo " Nodes       : $(sort -u $PBS_NODEFILE | tr '\n' ' ')"
echo " MPI ranks   : $NPROCS"
echo " Python      : $PYTHON"
echo " Grid N      : $ENDGAME_NX"
echo " Modes/NCV   : $ENDGAME_MODES / $ENDGAME_NCV"
echo " Output tag  : N${ENDGAME_NX}_q${ENDGAME_Q}_P${NPROCS}"
echo "========================================"

# Prevent thread oversubscription: each MPI rank should be single-threaded.
# Without this, OpenBLAS/MUMPS spawn extra threads and overload the cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Export ENDGAME_ params so main.py picks them up via os.environ
export ENDGAME_NX ENDGAME_Q ENDGAME_MODES ENDGAME_NCV ENDGAME_TARGET

mpiexec -n $NPROCS "$PYTHON" -u main.py
