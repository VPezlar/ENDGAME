#!/bin/bash
# MacBook Dev MPI Run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Initiating local MacBook MPI run across 8 cores..."

# Prevent OpenBLAS/MUMPS from spawning extra threads per MPI rank (oversubscription)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# -u: unbuffered stdout so progress prints appear in real time
mpiexec -n 8 python -u "$PROJECT_ROOT/main.py" < /dev/null
