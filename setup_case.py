#!/usr/bin/env python3
"""
Create an isolated case directory under ~/ENDGAME/cases/.

Each case has its own FDq/inputs/, FDq/output/, output/, logs/, and a
ready-to-submit PBS run.sh.

Key scheduling design:
  ppn=80 in the PBS directive reserves ALL physical cores on each node.
  This guarantees exclusive node access — no other job can co-locate.
  The actual MPI rank count (P) is hardcoded in the script; a temp
  hostfile with 16 ranks per node is constructed at runtime.

Usage:
    python setup_case.py Nx P [queue] [walltime_hours] [ppn] [imag_shift]

Examples:
    python setup_case.py 120 128 zeus_all_q 3 16 0.0
    python setup_case.py  43  16 zeus_all_q 2 16 0.0
"""
import os, sys, shutil

ROOT  = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(ROOT, 'cases')

Nx         = int(sys.argv[1])
P          = int(sys.argv[2])
queue      = sys.argv[3] if len(sys.argv) > 3 else 'zeus_all_q'
walltime_h = int(sys.argv[4]) if len(sys.argv) > 4 else 6
ppn        = int(sys.argv[5]) if len(sys.argv) > 5 else 16   # ranks per node
imag_shift = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
Ny         = int(sys.argv[7]) if len(sys.argv) > 7 else Nx
Nz         = int(sys.argv[8]) if len(sys.argv) > 8 else Nx
mumps_mem_mb = int(sys.argv[9]) if len(sys.argv) > 9 else 0

nodes = (P + ppn - 1) // ppn   # number of nodes needed

Q, MODES, NCV, TARGET = 6, 40, 100, 43.0

if Nx == Ny == Nz:
    tag = f"N{Nx}_q{Q}_P{P}"
else:
    tag = f"Nx{Nx}_Ny{Ny}_Nz{Nz}_q{Q}_P{P}"

case_dir = os.path.join(CASES, tag)

# Create directory structure
for d in ['FDq/inputs', 'FDq/output', 'output', 'logs']:
    os.makedirs(os.path.join(case_dir, d), exist_ok=True)

# main.py: copy so __file__ points into the case dir
shutil.copy2(os.path.join(ROOT, 'main.py'), os.path.join(case_dir, 'main.py'))

# src: symlink — shared read-only Python source
src_link = os.path.join(case_dir, 'src')
if not os.path.lexists(src_link):
    os.symlink(os.path.join(ROOT, 'src'), src_link)

# NY/NZ export lines
ny_export = f'export ENDGAME_NY={Ny}' if Ny != Nx else ''
nz_export = f'export ENDGAME_NZ={Nz}' if Nz != Nx else ''

nx_xflags = '    -x ENDGAME_NX -x ENDGAME_Q \\'
if Ny != Nx or Nz != Nx:
    nx_xflags = '    -x ENDGAME_NX -x ENDGAME_NY -x ENDGAME_NZ -x ENDGAME_Q \\'

run_lines = [
    '#!/bin/bash',
    f'#PBS -N ENDGAME_{tag}',
    f'#PBS -q {queue}',
    f'#PBS -l nodes={nodes}:ppn=80',
    f'#PBS -l walltime={walltime_h:02d}:00:00',
    '#PBS -j oe',
    '',
    '# --- Output routing -------------------------------------------------------',
    '# Solver progress (stdout) -> run.log   Warnings/errors (stderr) -> run.err',
    'mkdir -p "$PBS_O_WORKDIR/logs"',
    'exec > "$PBS_O_WORKDIR/logs/run.log" 2>"$PBS_O_WORKDIR/logs/run.err"',
    '',
    '# --- Node setup -----------------------------------------------------------',
    '# ppn=80 reserves the full node (80 physical cores) -> exclusive access.',
    '# We only launch ppn_mpi=16 MPI ranks per node; build a proper hostfile.',
    f'NPROCS={P}',
    f'PPN_MPI={ppn}',
    'PYTHON="$HOME/miniconda3/envs/tri_engine_complex/bin/python"',
    '',
    f'echo "ENDGAME case: {tag}  Job: $PBS_JOBID  Ranks: $NPROCS"',
    "echo \"Nodes: $(sort -u $PBS_NODEFILE | tr '\\n' ' ')\"",
    '',
    '# Build mpiexec hostfile: PPN_MPI entries per unique node',
    'MPI_HOSTFILE=$(mktemp /tmp/endgame_hosts.XXXXXX)',
    'sort -u "$PBS_NODEFILE" | while read _node; do',
    '    for _i in $(seq 1 $PPN_MPI); do echo "$_node"; done',
    'done > "$MPI_HOSTFILE"',
    '',
    '# --- Environment ----------------------------------------------------------',
    'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
    'export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx OMPI_MCA_btl=self,vader',
    'export UCX_LOG_LEVEL=error',
    f'export ENDGAME_NX={Nx}',
]
if ny_export:
    run_lines.append(ny_export)
if nz_export:
    run_lines.append(nz_export)
run_lines += [
    f'export ENDGAME_Q={Q}',
    f'export ENDGAME_MODES={MODES} ENDGAME_NCV={NCV} ENDGAME_TARGET={TARGET}',
    f'export ENDGAME_IMAG_SHIFT={imag_shift}',
    f'export ENDGAME_MUMPS_MEM_MB={mumps_mem_mb}',
    '',
    '# --- Launch ---------------------------------------------------------------',
    '$(dirname "$PYTHON")/mpiexec --prefix $(dirname $(dirname "$PYTHON")) \\',
    '    -n $NPROCS --bind-to none --hostfile "$MPI_HOSTFILE" \\',
    '    --mca btl self,vader \\',
    '    -x PYTHONDONTWRITEBYTECODE \\',
    '    -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS \\',
    nx_xflags,
    '    -x ENDGAME_MODES -x ENDGAME_NCV -x ENDGAME_TARGET \\',
    '    -x ENDGAME_IMAG_SHIFT \\',
    '    -x ENDGAME_MUMPS_MEM_MB \\',
    '    -x UCX_LOG_LEVEL \\',
    '    "$PYTHON" -u "$PBS_O_WORKDIR/main.py"',
    '',
    'rm -f "$MPI_HOSTFILE"',
]

run_path = os.path.join(case_dir, 'run.sh')
with open(run_path, 'w') as f:
    f.write('\n'.join(run_lines) + '\n')
os.chmod(run_path, 0o755)

N_total = (Nx+1)*(Ny+1)*(Nz+1)
grid_str = f"{Nx}x{Ny}x{Nz}" if not (Nx == Ny == Nz) else f"{Nx}^3"
print(f"Created : {case_dir}")
print(f"  Grid={grid_str}  DOF≈{N_total:,}  Nodes={nodes}  ppn_mpi={ppn}  ppn_pbs=80  ranks={P}")
print(f"Submit  : cd '{case_dir}' && qsub run.sh")
