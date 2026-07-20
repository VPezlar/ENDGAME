#!/bin/bash
# Generic single-run PBS script for the ENDGAME scaling study.
# Parameters are passed via -v on the qsub command line, e.g.:
#   qsub -v "ENDGAME_NX=60,ENDGAME_NODES=4,ENDGAME_PPN=16" run_scaling_job.sh
#
# Required env vars:  ENDGAME_NX, ENDGAME_NODES, ENDGAME_PPN
# Optional env vars:  ENDGAME_Q, ENDGAME_MODES, ENDGAME_NCV, ENDGAME_TARGET
#
# The #PBS directives below are placeholders; the real resource request
# is injected by the submission scripts using qsub -l.

#PBS -N ENDGAME_scaling
#PBS -q zeus_all_q
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o logs/scaling_pbs.log

cd $PBS_O_WORKDIR
mkdir -p logs

# Count total MPI ranks from the nodefile (PBS Pro compatible)
NPROCS=$(wc -l < $PBS_NODEFILE)

# Default params (overridden by environment when qsub -v is used)
: ${ENDGAME_NX:=30}
: ${ENDGAME_Q:=6}
: ${ENDGAME_MODES:=100}
: ${ENDGAME_NCV:=300}
: ${ENDGAME_TARGET:=43.0}

echo "========================================"
echo " ENDGAME scaling job"
echo "========================================"
echo " Job ID      : $PBS_JOBID"
echo " Nodes       : $(sort -u $PBS_NODEFILE | tr '\n' ' ')"
echo " MPI ranks   : $NPROCS"
echo " Grid N      : $ENDGAME_NX"
echo " Modes/NCV   : $ENDGAME_MODES / $ENDGAME_NCV"
echo " Output tag  : N${ENDGAME_NX}_q${ENDGAME_Q}_P${NPROCS}"
echo "========================================"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tri_engine

# Prevent thread oversubscription inside MPI ranks
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Export all ENDGAME_ variables so main.py picks them up
export ENDGAME_NX ENDGAME_Q ENDGAME_MODES ENDGAME_NCV ENDGAME_TARGET

mpiexec -n $NPROCS python -u main.py
