#!/bin/bash
#PBS -N ENDGAME_N54_q6_P32
#PBS -q zeus_all_q
#PBS -l nodes=2:ppn=80
#PBS -l walltime=02:00:00
#PBS -j oe

# --- Output routing -------------------------------------------------------
# Solver progress (stdout) -> run.log   Warnings/errors (stderr) -> run.err
mkdir -p "$PBS_O_WORKDIR/logs"
exec > "$PBS_O_WORKDIR/logs/run.log" 2>"$PBS_O_WORKDIR/logs/run.err"

# --- Node setup -----------------------------------------------------------
# ppn=80 reserves the full node (80 physical cores) -> exclusive access.
# We only launch ppn_mpi=16 MPI ranks per node; build a proper hostfile.
NPROCS=32
PPN_MPI=16
PYTHON="$HOME/miniconda3/envs/tri_engine_complex/bin/python"

echo "ENDGAME case: N54_q6_P32  Job: $PBS_JOBID  Ranks: $NPROCS"
echo "Nodes: $(sort -u $PBS_NODEFILE | tr '\n' ' ')"

# Build mpiexec hostfile: PPN_MPI entries per unique node
MPI_HOSTFILE=$(mktemp /tmp/endgame_hosts.XXXXXX)
sort -u "$PBS_NODEFILE" | while read _node; do
    for _i in $(seq 1 $PPN_MPI); do echo "$_node"; done
done > "$MPI_HOSTFILE"

# --- Environment ----------------------------------------------------------
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx OMPI_MCA_btl=self,vader
export UCX_LOG_LEVEL=error
export ENDGAME_NX=54
export ENDGAME_Q=6
export ENDGAME_MODES=40 ENDGAME_NCV=100 ENDGAME_TARGET=43.0
export ENDGAME_IMAG_SHIFT=0.0
export ENDGAME_MUMPS_MEM_MB=0

# --- Launch ---------------------------------------------------------------
$(dirname "$PYTHON")/mpiexec --prefix $(dirname $(dirname "$PYTHON")) \
    -n $NPROCS --bind-to none --hostfile "$MPI_HOSTFILE" \
    --mca btl self,vader \
    -x PYTHONDONTWRITEBYTECODE \
    -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS \
    -x ENDGAME_NX -x ENDGAME_Q \
    -x ENDGAME_MODES -x ENDGAME_NCV -x ENDGAME_TARGET \
    -x ENDGAME_IMAG_SHIFT \
    -x ENDGAME_MUMPS_MEM_MB \
    -x UCX_LOG_LEVEL \
    "$PYTHON" -u "$PBS_O_WORKDIR/main.py"

rm -f "$MPI_HOSTFILE"
