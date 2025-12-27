#!/usr/bin/env python3
"""
Parallel post-processing for UMA-screened ASE databases.

Features:
    (I)     Extract lowest-energy structures
    (II)    Plot UMA energy distribution
    (III)   Export metadata to CSV and JSON
    (IV)    Parallel export of POSCARs
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from ase.db import connect
from ase.io import write
import sqlite3
import os
import json
import multiprocessing as mp


# ============================================================
# Load Database
# ============================================================

def load_db(db_path):
    """Extract metadata (mlip_energy, poscar_source, etc.) from new ASE schema."""

    conn = sqlite3.connect(db_path)

    numkv = pd.read_sql_query(
        "SELECT key, value, id FROM number_key_values;",
        conn
    )
    textkv = pd.read_sql_query(
        "SELECT key, value, id FROM text_key_values;",
        conn
    )
    systems = pd.read_sql_query(
        "SELECT id FROM systems;",
        conn
    )

    num_pivot = numkv.pivot_table(index="id", columns="key", values="value", aggfunc="first")
    text_pivot = textkv.pivot_table(index="id", columns="key", values="value", aggfunc="first")

    df = systems.merge(num_pivot, left_on="id", right_index=True, how="left")
    df = df.merge(text_pivot, left_on="id", right_index=True, how="left")

    df = df.sort_values("mlip_energy")

    return df


# ============================================================
# Parallel POSCAR Writer
# ============================================================

def write_poscar_task(args):
    """Worker task for writing POSCARs in parallel."""
    db_path, sid, out_dir, fname = args

    with connect(db_path) as db:
        atoms = db.get_atoms(id=sid)

    write(fname, atoms, format="vasp", sort=True, direct=True, vasp5=True)
    return fname


# ============================================================
# Export lowest POSCARs (Parallel)
# ============================================================

def export_lowest(df, db_path, top_n=20, out_dir="lowest_poscars", workers=None):
    os.makedirs(out_dir, exist_ok=True)

    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    tasks = []
    for i, row in df.head(top_n).iterrows():
        sid = row["id"]
        fname = os.path.join(out_dir, f"POSCAR_lowest_{i:04d}")
        tasks.append((db_path, sid, out_dir, fname))

    print(f"[INFO] Exporting {len(tasks)} POSCARs using {workers} workers...")

    with mp.Pool(workers) as pool:
        for _ in pool.imap_unordered(write_poscar_task, tasks):
            pass

    print(f"Exported top {top_n} lowest-energy POSCARs to {out_dir}/")


# ============================================================
# Export POSCARs listed in JSON (Parallel)
# ============================================================

def export_poscars_from_json(json_file, db_path, out_dir="exported_poscars", workers=None):
    os.makedirs(out_dir, exist_ok=True)

    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    with open(json_file, "r") as f:
        data = json.load(f)

    tasks = []
    for entry in data:
        sid = entry["id"]
        fname = os.path.join(out_dir, f"POSCAR_id_{sid:04d}")
        tasks.append((db_path, sid, out_dir, fname))

    print(f"[INFO] Exporting {len(tasks)} POSCARs using {workers} workers...")

    with mp.Pool(workers) as pool:
        for _ in pool.imap_unordered(write_poscar_task, tasks):
            pass

    print(f"Exported {len(tasks)} POSCARs to {out_dir}/")


# ============================================================
# Plotting
# ============================================================

def plot_distribution(df, plot_file="energy_hist.png"):
    df_clean = df[df["mlip_energy"].notnull()]
    energies = df_clean["mlip_energy"].astype(float)

    plt.figure(figsize=(8, 5))
    plt.hist(energies, bins=50)
    plt.xlabel("UMA Energy (eV)")
    plt.ylabel("Count")
    plt.title("UMA Energy Distribution")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Saved energy distribution plot to {plot_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze UMA ASE database (parallel version)")
    parser.add_argument("db", help="Path to ASE database")
    parser.add_argument("--top", type=int, default=20, help="Number of lowest-energy structures to export")
    parser.add_argument("--csv", default="uma_results.csv", help="CSV output file")
    parser.add_argument("--json", default="uma_results.json", help="JSON output file")
    parser.add_argument("--plot", default="energy_hist.png", help="Histogram output file")
    parser.add_argument("--poscars", action="store_true", help="Export POSCARs listed in JSON output")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--energy-tol", type=float, default=1e-5, help="Energy tolerance for degeneracy filtering (default: 1e-5)")
    args = parser.parse_args()

    print(f"Loading database: {args.db}")
    df = load_db(args.db)
    total_strucs = len(df)
    df = df[df["mlip_energy"].notnull()].copy()
    df["mlip_energy"] = df["mlip_energy"].astype(float)
    df = df.sort_values("mlip_energy").reset_index(drop=True)

    # ============================================
    # Post-hoc degeneracy filtering
    # ============================================
    energy_tol = args.energy_tol

    unique_rows = []
    seen = []

    for _, row in df.iterrows():
        E = row["mlip_energy"]
        if not any(abs(E - e0) < energy_tol for e0 in seen):
            seen.append(E)
            unique_rows.append(row)

    df = pd.DataFrame(unique_rows).reset_index(drop=True)
    print(f"[INFO] Degeneracy filter: kept {len(df)} unique structures out of {total_strucs} (energy_tol={energy_tol})")


    df.to_csv(args.csv, index=False)
    df.to_json(args.json, orient="records", indent=2)
    print(f"Saved CSV to {args.csv}")
    print(f"Saved JSON to {args.json}")

    plot_distribution(df, args.plot)

    if args.poscars:
        export_poscars_from_json(args.json, args.db, workers=args.workers)
    else:
        export_lowest(df, args.db, top_n=args.top, workers=args.workers)

    print("Done.")


if __name__ == "__main__":
    mp.freeze_support()  # required for Windows / HPC spawn
    main()
