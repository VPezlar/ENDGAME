#!/usr/bin/env python3
"""
Create an isolated case directory under ~/ENDGAME/cases/.

Each case has its own FDq/inputs/, FDq/output/, output/, and a
ready-to-submit PBS run.sh. Concurrent cases never touch each other.

Usage:
    python setup_case.py Nx P [queue] [walltime_hours] [ppn] [imag_shift] [Ny] [Nz]

    Ny and Nz default to Nx (cubic grid). Specify them for anisotropic grids.

Examples:
    python setup_case.py 100 128 zeus_all_q 2 16 0.5        # cubic,  complex
    python setup_case.py 100 128 zeus_all_q 2 16 0.0 50 30  # Nx=100, Ny=50, Nz=30
    python setup_case.py  60  64 zeus_all_q 4 16 0.0        # cubic,  real
"""
import os, sys, shutil

ROOT  = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(ROOT, 'cases')

Nx         = int(sys.argv[1])
P          = int(sys.argv[2])
queue      = sys.argv[3] if len(sys.argv) > 3 else 'zeus_all_q'
walltime_h = int(sys.argv[4]) if len(sys.argv) > 4 else 6
ppn        = int(sys.argv[5]) if len(sys.argv) > 5 else 16
imag_shift = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
Ny         = int(sys.argv[7]) if len(sys.argv) > 7 else Nx
Nz         = int(sys.argv[8]) if len(sys.argv) > 8 else Nx
nodes      = (P + ppn - 1) // ppn

Q, MODES, NCV, TARGET = 6, 40, 100, 43.0

# Tag: compact for cubic, explicit for anisotropic
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

# Build NY/NZ export lines (only emit if different from NX to keep scripts clean)
ny_export = f'export ENDGAME_NY={Ny}' if Ny != Nx else ''
nz_export = f'export ENDGAME_NZ={Nz}' if Nz != Nx else ''

# Build -x propagation flags for NY/NZ
nx_xflags = '    -x ENDGAME_NX -x ENDGAME_Q \\'
if Ny != Nx or Nz != Nx:
    nx_xflags = '    -x ENDGAME_NX -x ENDGAME_NY -x ENDGAME_NZ -x ENDGAME_Q \\'

# Write run.sh
run_lines = [
    '#!/bin/bash',
    f'#PBS -N ENDGAME_{tag}',
    f'#PBS -q {queue}',
    f'#PBS -l nodes={nodes}:ppn={ppn}',
    f'#PBS -l walltime={walltime_h:02d}:00:00',
    f'#PBS -l pvmem=16gb',
    '#PBS -j oe',
    '',
    '# Redirect output immediately so tail -f works in real time.',
    'mkdir -p "$PBS_O_WORKDIR/logs"',
    'exec > "$PBS_O_WORKDIR/logs/run.log" 2>&1',
    '',
    'NPROCS=$(wc -l < $PBS_NODEFILE)',
    'PYTHON="$HOME/miniconda3/envs/tri_engine_complex/bin/python"',
    '',
    f'echo "ENDGAME case: {tag}  Job: $PBS_JOBID  Ranks: $NPROCS"',
    'echo "Nodes: $(sort -u $PBS_NODEFILE | tr \'\\n\' \' \')"',
    '',
    'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
    '# Route MPI through UCX/InfiniBand (avoids TCP BTL multi-NIC confusion)',
    'export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx OMPI_MCA_btl=self,vader',
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
    '',
    '$(dirname "$PYTHON")/mpiexec --prefix $(dirname $(dirname "$PYTHON")) \\',
    '    -n $NPROCS --bind-to none --hostfile $PBS_NODEFILE \\',
    '    --mca btl self,vader \\',
    '    -x PYTHONDONTWRITEBYTECODE \\',
    '    -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS \\',
    nx_xflags,
    '    -x ENDGAME_MODES -x ENDGAME_NCV -x ENDGAME_TARGET \\',
    '    -x ENDGAME_IMAG_SHIFT \\',
    '    "$PYTHON" -u "$PBS_O_WORKDIR/main.py"',
]

run_path = os.path.join(case_dir, 'run.sh')
with open(run_path, 'w') as f:
    f.write('\n'.join(run_lines) + '\n')
os.chmod(run_path, 0o755)

# WARNING — Stale NFS file handle
# Once a job has been submitted from this case directory, do NOT
# delete or recreate it while the job is running. On NFS, deleting
# a directory removes its inode. The running job holds NFS handles
# to the old inode; if you rm -rf and recreate, any writes at job
# end will get errno 116 "Stale file handle" and the results will
# be lost. Each case directory is permanent for the lifetime of its job.

N_total = (Nx+1)*(Ny+1)*(Nz+1)
grid_str = f"{Nx}x{Ny}x{Nz}" if not (Nx == Ny == Nz) else f"{Nx}^3"
print(f"Created : {case_dir}")
print(f"  Grid={grid_str}  DOF≈{N_total:,}  Nodes={nodes}  ppn={ppn}  ranks={P}  NCV={NCV}  modes={MODES}")
if imag_shift:
    print(f"  Imag shift : {imag_shift}  ← complex eigenvalues active")
print(f"Submit  : cd '{case_dir}' && qsub run.sh")
