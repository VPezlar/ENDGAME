#!/bin/bash
# Strong scaling study: fixed problem size N=60, vary MPI ranks.
#
# N=60  ->  DOF = (60+1)^3 = 226,981
# q=6, num_modes=100, krylov_size=300, target_metric=43
#
# P and node layout:
#   P=16  -> 1 node  x 16 ppn
#   P=32  -> 2 nodes x 16 ppn
#   P=64  -> 4 nodes x 16 ppn
#   P=128 -> 8 nodes x 16 ppn
#   P=192 -> 12 nodes x 16 ppn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
mkdir -p logs

N=60
Q=6
MODES=100
NCV=300
TARGET=43.0
WALLTIME="06:00:00"
QUEUE="zeus_all_q"

# (nodes, ppn) pairs for each P
declare -a NODES_LIST=(1  2  4  8  12)
declare -a PPN_LIST=(  16 16 16 16 16)

echo "Submitting STRONG scaling study: N=$N, P = 16 32 64 128 192"
echo ""

for idx in 0 1 2 3 4; do
    NODES=${NODES_LIST[$idx]}
    PPN=${PPN_LIST[$idx]}
    P=$(( NODES * PPN ))
    TAG="N${N}_q${Q}_P${P}"
    LOGFILE="logs/strong_${TAG}.log"

    JOB_ID=$(qsub \
        -N "ENDGAME_${TAG}" \
        -q "$QUEUE" \
        -l "nodes=${NODES}:ppn=${PPN}" \
        -l "walltime=${WALLTIME}" \
        -j oe \
        -o "$LOGFILE" \
        -v "ENDGAME_NX=${N},ENDGAME_Q=${Q},ENDGAME_MODES=${MODES},ENDGAME_NCV=${NCV},ENDGAME_TARGET=${TARGET}" \
        jobs/run_scaling_job.sh)

    echo "  P=${P}  (${NODES}n x ${PPN}ppn)  tag=${TAG}  job=${JOB_ID}"
done

echo ""
echo "All strong-scaling jobs submitted. Monitor with: qstat -u vojtech-p"
echo "After completion: python collect_timing.py output/N${N}_*"
