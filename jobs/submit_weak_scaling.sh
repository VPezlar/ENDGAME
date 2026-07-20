#!/bin/bash
# Weak scaling study: DOF/rank ~ 4300 constant across all runs.
# Problem size grows proportionally with P => ideal time constant.
#
# (P,  N)   DOF        DOF/rank
# (16, 40)  68,921     4,308
# (32, 50)  132,651    4,145
# (64, 64)  274,625    4,291
# (128,81)  551,368    4,308
# (192,92)  804,357    4,189

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
mkdir -p logs

Q=6
MODES=100
NCV=300
TARGET=43.0
WALLTIME="12:00:00"
QUEUE="zeus_all_q"

# Parallel to (P, N, NODES, PPN)
declare -a P_LIST=(    16  32  64  128 192)
declare -a N_LIST=(    40  50  64  81  92)
declare -a NODES_LIST=(1   2   4   8   12)
declare -a PPN_LIST=(  16  16  16  16  16)

echo "Submitting WEAK scaling study: DOF/rank ~ 4300"
echo ""

for idx in 0 1 2 3 4; do
    P=${P_LIST[$idx]}
    N=${N_LIST[$idx]}
    NODES=${NODES_LIST[$idx]}
    PPN=${PPN_LIST[$idx]}
    TAG="N${N}_q${Q}_P${P}"
    LOGFILE="logs/weak_${TAG}.log"

    JOB_ID=$(qsub \
        -N "ENDGAME_${TAG}" \
        -q "$QUEUE" \
        -l "nodes=${NODES}:ppn=${PPN}" \
        -l "walltime=${WALLTIME}" \
        -j oe \
        -o "$LOGFILE" \
        -v "ENDGAME_NX=${N},ENDGAME_Q=${Q},ENDGAME_MODES=${MODES},ENDGAME_NCV=${NCV},ENDGAME_TARGET=${TARGET}" \
        jobs/run_scaling_job.sh)

    DOF=$(python3 -c "print((${N}+1)**3)")
    echo "  P=${P}  N=${N}  DOF=${DOF}  (${NODES}n x ${PPN}ppn)  job=${JOB_ID}"
done

echo ""
echo "All weak-scaling jobs submitted. Monitor with: qstat -u vojtech-p"
echo "After completion: python collect_timing.py"
