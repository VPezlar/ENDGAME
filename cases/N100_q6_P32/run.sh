#!/bin/bash
#PBS -N ENDGAME_N100_q6_P32
#PBS -q zeus_all_q
#PBS -l nodes=8:ppn=4
#PBS -l walltime=03:00:00
#PBS -l pvmem=16gb
#PBS -j oe

# Redirect output immediately so tail -f works in real time.
mkdir -p "$PBS_O_WORKDIR/logs"
exec > "$PBS_O_WORKDIR/logs/run.log" 2>&1

NPROCS=$(wc -l < $PBS_NODEFILE)
PYTHON="$HOME/miniconda3/envs/tri_engine_complex/bin/python"

echo "ENDGAME case: N100_q6_P32  Job: $PBS_JOBID  Ranks: $NPROCS"
echo "Nodes: $(sort -u $PBS_NODEFILE | tr '\n' ' ')"

export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Route MPI through UCX/InfiniBand (avoids TCP BTL multi-NIC confusion)
export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx OMPI_MCA_btl=self,vader
export ENDGAME_NX=100
export ENDGAME_Q=6
export ENDGAME_MODES=40 ENDGAME_NCV=100 ENDGAME_TARGET=43.0
export ENDGAME_IMAG_SHIFT=0.0
export ENDGAME_MUMPS_MEM_MB=12500

$(dirname "$PYTHON")/mpiexec --prefix $(dirname $(dirname "$PYTHON")) \
    -n $NPROCS --bind-to none --hostfile $PBS_NODEFILE \
    --mca btl self,vader \
    -x PYTHONDONTWRITEBYTECODE \
    -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS \
    -x ENDGAME_NX -x ENDGAME_Q \
    -x ENDGAME_MODES -x ENDGAME_NCV -x ENDGAME_TARGET \
    -x ENDGAME_IMAG_SHIFT \
    -x ENDGAME_MUMPS_MEM_MB \
    "$PYTHON" -u "$PBS_O_WORKDIR/main.py"
