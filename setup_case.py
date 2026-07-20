#!/usr/bin/env python3
"""
Create an isolated case directory under ~/ENDGAME/cases/.

Each case has its own FDq/inputs/, FDq/output/, output/, and a
ready-to-submit PBS run.sh. Concurrent cases never touch each other.

Usage:
    python setup_case.py N P [queue] [walltime_hours]

Example — strong scaling:
    python setup_case.py 60 16  zeus_all_q 6
    python setup_case.py 60 32  zeus_all_q 6
    python setup_case.py 60 64  zeus_all_q 6
    python setup_case.py 60 128 zeus_all_q 6
    python setup_case.py 60 192 zeus_all_q 6
"""
import os, sys, shutil

ROOT  = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(ROOT, 'cases')

N          = int(sys.argv[1])
P          = int(sys.argv[2])
queue      = sys.argv[3] if len(sys.argv) > 3 else 'zeus_all_q'
walltime_h = int(sys.argv[4]) if len(sys.argv) > 4 else 6
ppn        = int(sys.argv[5]) if len(sys.argv) > 5 else 64
nodes      = (P + ppn - 1) // ppn

Q, MODES, NCV, TARGET = 6, 100, 300, 43.0
tag      = f"N{N}_q{Q}_P{P}"
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

# Write run.sh
run_lines = [
    '#!/bin/bash',
    f'#PBS -N ENDGAME_{tag}',
    f'#PBS -q {queue}',
    f'#PBS -l nodes={nodes}:ppn={ppn}',
    f'#PBS -l walltime={walltime_h:02d}:00:00',
    '#PBS -j oe',
    '#PBS -o logs/run.log',
    '',
    '# PBS_O_WORKDIR = this case directory (qsub submitted from here)',
    'NPROCS=$(wc -l < $PBS_NODEFILE)',
    'PYTHON="$HOME/miniconda3/envs/tri_engine/bin/python"',
    '',
    f'echo "ENDGAME case: {tag}  Job: $PBS_JOBID  Ranks: $NPROCS"',
    '',
    'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
    f'export ENDGAME_NX={N} ENDGAME_Q={Q}',
    f'export ENDGAME_MODES={MODES} ENDGAME_NCV={NCV} ENDGAME_TARGET={TARGET}',
    '',
    'mpiexec -n $NPROCS --bind-to none \\',
    '    -x PYTHONDONTWRITEBYTECODE \\',
    '    -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS \\',
    '    -x ENDGAME_NX -x ENDGAME_Q \\',
    '    -x ENDGAME_MODES -x ENDGAME_NCV -x ENDGAME_TARGET \\',
    '    "$PYTHON" -u "$PBS_O_WORKDIR/main.py"',
]

run_path = os.path.join(case_dir, 'run.sh')
with open(run_path, 'w') as f:
    f.write('\n'.join(run_lines) + '\n')
os.chmod(run_path, 0o755)

print(f"Created: {case_dir}")
print(f"Submit:  cd '{case_dir}' && qsub run.sh")
