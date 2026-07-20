"""
collect_timing.py — Aggregate timing.json files from all scaling runs
and print a formatted table to stdout.

Usage:
    python collect_timing.py              # scan all output/*/timing.json
    python collect_timing.py output/N60*  # only strong-scaling results

Columns:
    run_tag | N | P | DOF | DOF/rank | nnz_A | t_assemble | t_mumps | t_krylov | t_total | nconv
"""
import os
import sys
import json
import glob


def load_timings(pattern):
    files = sorted(glob.glob(pattern))
    records = []
    for f in files:
        try:
            with open(f) as fh:
                records.append(json.load(fh))
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}", file=sys.stderr)
    return records


def print_table(records, title):
    if not records:
        print(f"\n{title}: no data found.\n")
        return

    # Sort by (N, P)
    records = sorted(records, key=lambda r: (r["Nx"], r["mpi_ranks"]))

    hdr = (f"{'Run tag':<22} {'N':>4} {'P':>4} {'DOF':>10} "
           f"{'DOF/rank':>9} {'nnz_A':>10} "
           f"{'t_asmb':>8} {'t_mumps':>8} {'t_krylov':>9} {'t_total':>8} {'nconv':>6}")
    sep = "-" * len(hdr)
    print(f"\n{title}")
    print(sep)
    print(hdr)
    print(sep)

    for r in records:
        P   = r["mpi_ranks"]
        dof = r["N_total"]
        print(
            f"{r['run_tag']:<22} {r['Nx']:>4} {P:>4} {dof:>10,} "
            f"{dof//P:>9,} {r.get('nnz_A',0):>10,} "
            f"{r.get('t_assemble_s',-1):>8.2f} "
            f"{r.get('t_mumps_s',-1):>8.2f} "
            f"{r.get('t_krylov_s',-1):>9.2f} "
            f"{r.get('t_total_s',-1):>8.2f} "
            f"{r.get('nconv',-1):>6}"
        )
    print(sep)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User provided glob patterns
        pattern_list = sys.argv[1:]
        records = []
        for pat in pattern_list:
            if os.path.isdir(pat):
                records += load_timings(os.path.join(pat, "timing.json"))
            else:
                records += load_timings(pat)
        print_table(records, "Custom selection")
    else:
        # Auto-detect strong vs weak from run_tag naming
        all_records = load_timings("output/*/timing.json")

        # Strong scaling: same N across different P values
        from collections import defaultdict
        by_N = defaultdict(list)
        for r in all_records:
            by_N[r["Nx"]].append(r)

        # Runs where the same N appears with multiple P are strong scaling
        strong = []
        weak_all = []
        for N, recs in by_N.items():
            if len(recs) > 1:
                strong += recs
            else:
                weak_all += recs

        # Runs where N varies across a fixed DOF/rank are weak scaling
        # (heuristic: if same N appears only once, it is likely weak scaling)
        # Just print both groups separately
        print_table(strong,   "STRONG SCALING  (fixed N, vary P)")
        print_table(weak_all, "WEAK SCALING    (vary N with P, DOF/rank ~ const)")
        print_table(all_records, "ALL RUNS COMBINED")
