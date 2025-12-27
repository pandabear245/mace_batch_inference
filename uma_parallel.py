#!/usr/bin/env python3
"""
Parallel intralayer permutation generator + UMA-S-1p1 screening.

- True permutations of cations within each layer (no repetition, composition preserved per layer)
- Multiple CPU workers generate structures in parallel
- One GPU worker batches them, runs UMA-S-1p1, screens, and writes to ASE .db
- Energy-degeneracy filtering: skip structures with identical UMA energy (within tolerance)
- --count mode prints both:
    * with repetition  : n^n per layer, product over layers
    * without repetition: n!  per layer, product over layers
"""

import sys
import os
import math
import argparse
#from itertools import product
from itertools import permutations
import multiprocessing as mp

import numpy as np
from ase import Atoms
from ase.db import connect


# ============================================================
#   POSCAR HELPERS
# ============================================================

def extract_lattice_vectors(poscar_path):
    with open(poscar_path, "r") as f:
        lines = f.readlines()
    lattice_vectors = [list(map(float, lines[i].split())) for i in range(2, 5)]
    return np.array(lattice_vectors)


def extract_positions(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Direct") or line.strip().startswith("Cartesian"):
            position_start_line = i + 1
            break
    else:
        raise ValueError("No 'Direct' or 'Cartesian' line found.")
    positions = []
    for line in lines[position_start_line:]:
        tokens = line.strip().split()
        if len(tokens) >= 3:
            positions.append(list(map(float, tokens[:3])))
        else:
            break
    return np.array(positions)


def read_poscar_symbols(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Direct") or line.strip().startswith("Cartesian"):
            start = i + 1
            break
    else:
        raise ValueError("No 'Direct' or 'Cartesian' line found.")
    symbols = []
    for line in lines[start:]:
        tokens = line.strip().split()
        if len(tokens) >= 4:
            symbols.append(tokens[3])
        else:
            break
    return symbols


def group_atoms_by_layer(positions, symbols, axis_index, tol=1e-3):
    """
    Group atoms into layers based on axis_index coordinate.
    """
    combined = sorted(zip(positions, symbols), key=lambda x: x[0][axis_index])
    layers = []
    current_layer = []
    current_coord = None

    for pos, sym in combined:
        coord = pos[axis_index]
        if current_coord is None or abs(coord - current_coord) < tol:
            current_layer.append((pos, sym))
            current_coord = coord
        else:
            layers.append(current_layer)
            current_layer = [(pos, sym)]
            current_coord = coord

    if current_layer:
        layers.append(current_layer)

    return layers


# ============================================================
#   PER-LAYER PERMUTATIONS (NO REPETITION)
# ============================================================


def generate_intralayer_products(layers):
    """
    For each layer:
      - identify cation sites (non-anions)
      - generate TRUE permutations (no repetition) of cations across these sites
      - anions stay fixed

    Returns:
      list of layers_configs, where each element is a list of configs for that layer.
      A single config is a list of (pos, sym) for that layer.
    """
    anions = {"C", "N", "O", "B"}
    all_layer_products = []

    for layer in layers:
        positions, symbols = zip(*layer)
        cation_indices = [i for i, s in enumerate(symbols) if s not in anions]
        cations = [symbols[i] for i in cation_indices]

        # true permutations (no repetition)
        #perms = product(cations, len(cation_indices)) # allowing for repetition
        perms = permutations(cations, len(cation_indices))
        layer_configs = []

        for p in perms:
            p = list(p)
            cfg = []
            idx = 0
            for i in range(len(symbols)):
                if i in cation_indices:
                    cfg.append((positions[i], p[idx]))
                    idx += 1
                else:
                    cfg.append((positions[i], symbols[i]))
            layer_configs.append(cfg)

        all_layer_products.append(layer_configs)

    return all_layer_products


def combo_to_atoms(combo, lattice, poscar_source):
    """
    Flatten layers → ASE Atoms.
    combo: list[layer_config], each layer_config = list[(pos, sym)]
    """
    flat = [atom for layer in combo for atom in layer]
    positions, symbols = zip(*flat)
    symbols_cleaned = [s.replace("_pv", "").replace("_sv", "") for s in symbols]

    # Unique species list preserving first appearance order (VASP-required)
    species_order = []
    for s in symbols_cleaned:
        if s not in species_order:
            species_order.append(s)

    atoms = Atoms(
        symbols_cleaned,
        #positions=np.array(positions),
        cell=lattice,
        pbc=True,
        #species=species_order,
    )
    # Set fractional positions (Direct)
    atoms.set_scaled_positions(np.array(positions))
    atoms.info["poscar_source"] = poscar_source
    return atoms


# ============================================================
#   COUNTING FUNCTION (BOTH MODES)
# ============================================================

def count_intralayer_permutations(poscar_file, axis="z"):
    """
    For each layer:
        n = number of cation sites
        with repetition   : n^n
        without repetition: n!
    Total = product over layers.

    Returns:
        (with_repetition_total, without_repetition_total)
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_index = axis_map[axis]

    positions = extract_positions(poscar_file)
    symbols = read_poscar_symbols(poscar_file)
    layers = group_atoms_by_layer(positions, symbols, axis_index)

    anions = {"C", "N", "O", "B"}
    total_rep = 1
    total_norep = 1

    for layer in layers:
        _, syms = zip(*layer)
        cations = [s for s in syms if s not in anions]
        n = len(cations)
        total_rep *= n ** n
        total_norep *= math.factorial(n)

    return total_rep, total_norep


# ============================================================
#   MIXED-RADIX INDEXING FOR PARALLEL GENERATION
# ============================================================

def flat_index_to_indices(flat_idx, bases):
    """
    Convert a flat index into a tuple of indices for each layer
    given the layer lengths (mixed-radix).

    bases: [len(layer0), len(layer1), ...]
    Returns: [i0, i1, ...] with 0 <= ik < bases[k].
    """
    idxs = []
    for base in reversed(bases):
        idxs.append(flat_idx % base)
        flat_idx //= base
    return list(reversed(idxs))


# ============================================================
#   PRODUCER (CPU WORKERS)
# ============================================================

def producer_worker(worker_id, n_workers, permuted_layers, lattice, poscar_file, queue):
    """
    CPU worker:
      - iterates over a subset of all combination indices
      - builds Atoms objects
      - sends them to the GPU worker via queue
    """
    bases = [len(layer) for layer in permuted_layers]
    total = 1
    for b in bases:
        total *= b

    for flat_idx in range(worker_id, total, n_workers):
        combo_indices = flat_index_to_indices(flat_idx, bases)
        combo = [
            permuted_layers[layer_i][combo_indices[layer_i]]
            for layer_i in range(len(permuted_layers))
        ]
        atoms = combo_to_atoms(combo, lattice, poscar_source=os.path.basename(poscar_file))
        queue.put(atoms)

    # signal termination from this worker
    queue.put(None)


# ============================================================
#   GPU WORKER (UMA-S-1p1 + SCREENING + DB)
# ============================================================

def gpu_worker(queue, db_path, num_producers, batch_size, delta_e, keep_frac, energy_tol, all_mode):
    """
    Single GPU worker:
      - receives Atoms from queue
      - batches them
      - runs UMA-S-1p1
      - screens + filters degenerates
      - writes accepted structures to ASE .db
    """
    import torch
    from fairchem.core import pretrained_mlip
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GPU] Using UMA-S-1p1 on device: {device}")

    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    db = connect(db_path)

    E_min = float("inf")
    seen_energies = []  # store floats
    producers_done = 0
    processed = 0
    accepted = 0

    atoms_buffer = []

    def process_buffer(buffer):
        nonlocal E_min, accepted, processed, seen_energies

        if not buffer:
            return

        # Build AtomicData
        atomic_data_list = [AtomicData.from_ase(a, task_name="omat") for a in buffer]

        try:
            batch = atomicdata_list_to_batch(atomic_data_list).to(device)
            with torch.no_grad():
                preds = predictor.predict(batch)
            energies = preds["energy"].detach().cpu().numpy()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("[GPU] CUDA OOM on batch, splitting...")
                torch.cuda.empty_cache()
                mid = max(1, len(atomic_data_list) // 2)
                process_buffer(buffer[:mid])
                process_buffer(buffer[mid:])
                return
            else:
                print(f"[GPU] RuntimeError during UMA prediction: {e}")
                return

        # Screening
        for atoms, E in zip(buffer, energies):
            processed += 1
            E = float(E)

            # Update global energy minimum
            if E < E_min:
                E_min = E

            #if any(abs(E-e0) < energy_tol for e0 in seen_energies):
            #    continue

            if all_mode:
                # Keep everything except degenerates
                accept = True
            else:
                # Original screening rule
                accept = False
                if any(abs(E-e0) < energy_tol for e0 in seen_energies):
                    continue
                if E <= E_min + delta_e:
                    accept = True
                elif np.random.random() < keep_frac:
                    accept = True

            if accept:
                seen_energies.append(E)
                db.write(
                    atoms,
                    poscar_source=atoms.info.get("poscar_source", ""),
                    mlip_model="uma-s-1p1",
                    mlip_energy=E,
                )
                accepted += 1

    while True:
        item = queue.get()
        if item is None:
            producers_done += 1
            if producers_done == num_producers:
                # no more incoming items
                break
            else:
                continue

        atoms_buffer.append(item)

        if len(atoms_buffer) >= batch_size:
            process_buffer(atoms_buffer)
            atoms_buffer = []

    # flush leftovers
    if atoms_buffer:
        process_buffer(atoms_buffer)

    print(f"[GPU] Done. Processed {processed} structures, accepted {accepted}, written to {db_path}")


# ============================================================
#   CLI + MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel intralayer permutation generator + UMA-S-1p1 screening"
    )
    p.add_argument("poscar", help="Input POSCAR file")
    p.add_argument(
        "--axis", choices=["x", "y", "z"], default="z",
        help="Axis along which to define layers (default: z)",
    )
    p.add_argument(
        "--batch-size", type=int, default=512,
        help="UMA batch size on GPU (default: 512)",
    )
    p.add_argument(
        "--delta-e", type=float, default=0.5,
        help="Energy window (eV) above current min that is always kept (default: 0.5)",
    )
    p.add_argument(
        "--keep-frac", type=float, default=0.002,
        help="Random retention fraction for high-energy structures (default: 0.002)",
    )
    p.add_argument(
        "--energy-tol", type=float, default=1e-5,
        help="Degeneracy energy tolerance in eV (default: 1e-5)",
    )
    p.add_argument(
        "--db-path", default="database/structures_uma_parallel.db",
        help="Path to ASE .db for accepted structures",
    )
    p.add_argument(
        "--workers", type=int, default=8,
        help="Number of CPU producer workers (default: 4)",
    )
    p.add_argument(
        "--count", action="store_true",
        help="Only print permutation counts (with & without repetition), then exit",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Keep ALL non-degenerate structures (disable energy window and random retention).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    poscar_file = args.poscar
    axis = args.axis

    if args.count:
        rep, norep = count_intralayer_permutations(poscar_file, axis=axis)
        print(f"With repetition (n^n):       {rep:,}")
        print(f"Without repetition (n!):     {norep:,}")
        return

    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_index = axis_map[axis]

    lattice = extract_lattice_vectors(poscar_file)
    positions = extract_positions(poscar_file)
    symbols = read_poscar_symbols(poscar_file)
    layers = group_atoms_by_layer(positions, symbols, axis_index)

    # generate per-layer permutation sets (true permutations)
    permuted_layers = generate_intralayer_products(layers)
    bases = [len(layer) for layer in permuted_layers]
    total = 1
    for b in bases:
        total *= b

    print(f"[INFO] Per-layer config counts: {bases}")
    print(f"[INFO] Total true permutations (no repetition): {total:,}")

    # multiprocessing setup
    num_producers = max(1, args.workers)
    print(f"[INFO] Starting {num_producers} CPU workers + 1 GPU worker")

    ctx = mp.get_context("spawn")  # safer with CUDA
    queue = ctx.Queue(maxsize=4 * args.batch_size)

    # start GPU worker
    gpu_proc = ctx.Process(
        target=gpu_worker,
        args=(queue, args.db_path, num_producers,
              args.batch_size, args.delta_e, args.keep_frac, args.energy_tol, args.all),
    )
    gpu_proc.start()

    # start producer workers
    producers = []
    for wid in range(num_producers):
        p = ctx.Process(
            target=producer_worker,
            args=(wid, num_producers, permuted_layers, lattice, poscar_file, queue),
        )
        p.start()
        producers.append(p)

    # wait for producers to finish
    for p in producers:
        p.join()

    # wait for GPU worker to finish
    gpu_proc.join()

    print("[MAIN] All processes finished.")


if __name__ == "__main__":
    base = "database"

    # Always start from database_01
    counter = 1
    while True:
        db_dir = f"{base}_{counter:02d}"
        if not os.path.exists(db_dir):
            break
        counter += 1

    os.makedirs(db_dir, exist_ok=True)
    print(f"[INFO] Using database directory: {db_dir}")

    # Only supply db-path if user did not specify one manually.
    # If user passes --db-path, we do NOT override it.
    if not any(arg.startswith("--db-path") for arg in sys.argv):
        sys.argv.append(f"--db-path={db_dir}/structures_uma_parallel.db")

    main()
