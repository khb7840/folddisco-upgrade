#!/usr/bin/env python3
"""Validate structural integrity of True/False sets via TM-score and RMSD sampling."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("[ERROR] numpy is required for validate_sets.py") from exc

try:
    from Bio.PDB import PDBParser, PPBuilder, Superimposer
except Exception as exc:  # pragma: no cover
    raise SystemExit("[ERROR] biopython is required for validate_sets.py") from exc


Record = Tuple[str, str, str, str]
Pair = Tuple[str, str]


def read_domain_list(path: Path) -> List[Record]:
    rows: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("ID"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


def cath_key(row: Record) -> str:
    return f"{row[1]}_{row[2]}_{row[3]}"


def choose_chain_id_from_domain_id(domain_id: str) -> Optional[str]:
    if len(domain_id) >= 5:
        return domain_id[4]
    return None


def extract_ca_coords(pdb_path: Path, domain_id: str) -> np.ndarray:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(domain_id, str(pdb_path))
    model = next(structure.get_models())

    preferred_chain = choose_chain_id_from_domain_id(domain_id)
    chain = None
    if preferred_chain and preferred_chain in model:
        chain = model[preferred_chain]
    else:
        chains = list(model.get_chains())
        if not chains:
            return np.empty((0, 3), dtype=float)
        chain = max(chains, key=lambda c: sum(1 for _ in c.get_residues()))

    coords = []
    for residue in chain:
        hetflag, _, _ = residue.id
        if hetflag != " ":
            continue
        if "CA" in residue:
            coords.append(residue["CA"].coord)

    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.array(coords, dtype=float)


def kabsch_superpose(ref: np.ndarray, mob: np.ndarray) -> Tuple[np.ndarray, float]:
    ref_center = ref.mean(axis=0)
    mob_center = mob.mean(axis=0)

    ref0 = ref - ref_center
    mob0 = mob - mob_center

    h = mob0.T @ ref0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    mob_aligned = (mob0 @ r) + ref_center
    diff = ref - mob_aligned
    rmsd = float(np.sqrt((diff * diff).sum() / len(ref)))
    return mob_aligned, rmsd


def tm_score_from_distances(distances: np.ndarray, length_norm: int) -> float:
    l = max(length_norm, 1)
    if l <= 15:
        d0 = 0.5
    else:
        d0 = 1.24 * ((l - 15) ** (1.0 / 3.0)) - 1.8
        d0 = max(d0, 0.5)
    return float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))


def fallback_pair_metrics(pdb_a: Path, pdb_b: Path, id_a: str, id_b: str) -> Optional[Tuple[float, float, str]]:
    ca_a = extract_ca_coords(pdb_a, id_a)
    ca_b = extract_ca_coords(pdb_b, id_b)
    if len(ca_a) < 5 or len(ca_b) < 5:
        return None

    n = min(len(ca_a), len(ca_b))
    ref = ca_a[:n]
    mob = ca_b[:n]
    mob_aligned, rmsd = kabsch_superpose(ref, mob)
    dists = np.linalg.norm(ref - mob_aligned, axis=1)
    tm = tm_score_from_distances(dists, n)
    return tm, rmsd, "fallback_kabsch"


def run_tmalign(pdb_a: Path, pdb_b: Path, tmalign_bin: str) -> Optional[Tuple[float, float, str]]:
    cmd = [tmalign_bin, str(pdb_a), str(pdb_b)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None

    text = proc.stdout
    tm_vals = [float(x) for x in re.findall(r"TM-score=\s*([0-9]*\.?[0-9]+)", text)]
    rmsd_match = re.search(r"RMSD=\s*([0-9]*\.?[0-9]+)", text)
    if not tm_vals or rmsd_match is None:
        return None

    tm = max(tm_vals)
    rmsd = float(rmsd_match.group(1))
    return tm, rmsd, "tmalign"


def build_true_pairs(rows_index: List[Record], existing_index_ids: set[str], sample_size: int, rng: random.Random) -> List[Pair]:
    by_sf: Dict[str, List[str]] = defaultdict(list)
    for row in rows_index:
        pid = row[0]
        if pid in existing_index_ids:
            by_sf[cath_key(row)].append(pid)

    all_pairs: List[Pair] = []
    for ids in by_sf.values():
        if len(ids) < 2:
            continue
        ids = sorted(set(ids))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                all_pairs.append((ids[i], ids[j]))

    if not all_pairs:
        return []
    if len(all_pairs) <= sample_size:
        rng.shuffle(all_pairs)
        return all_pairs
    return rng.sample(all_pairs, sample_size)


def build_false_pairs(
    rows_index: List[Record],
    rows_null: List[Record],
    existing_index_ids: set[str],
    existing_null_ids: set[str],
    sample_size: int,
    rng: random.Random,
) -> List[Pair]:
    index_sf = {row[0]: cath_key(row) for row in rows_index if row[0] in existing_index_ids}
    null_sf = {row[0]: cath_key(row) for row in rows_null if row[0] in existing_null_ids}

    null_ids = sorted(null_sf)
    index_ids = sorted(index_sf)

    pairs: List[Pair] = []
    tries = 0
    max_tries = sample_size * 100
    while len(pairs) < sample_size and tries < max_tries and null_ids and index_ids:
        tries += 1
        a = rng.choice(null_ids)
        b = rng.choice(index_ids)
        if a == b:
            continue
        if null_sf[a] == index_sf[b]:
            continue
        pairs.append((a, b))

    return pairs


def plot_tm_distribution(out_png: Path, true_scores: List[float], false_scores: List[float]) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib is not installed; plot generation skipped"

    if not true_scores and not false_scores:
        return "no TM-score values available for plotting"

    bins = np.linspace(0.0, 1.0, 31)
    plt.figure(figsize=(9, 5))
    if true_scores:
        plt.hist(true_scores, bins=bins, alpha=0.6, density=True, label="True set", color="#1f77b4")
    if false_scores:
        plt.hist(false_scores, bins=bins, alpha=0.6, density=True, label="False set", color="#d62728")
    plt.axvline(0.5, linestyle="--", color="black", linewidth=1.0, label="TM-score 0.5")
    plt.xlabel("TM-score")
    plt.ylabel("Density")
    plt.title("TM-score Distribution: True vs False Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    return None


def main() -> None:
    repo_root_default = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description="Validate structural integrity of true/false sets")
    ap.add_argument("--repo-root", type=Path, default=repo_root_default)
    ap.add_argument("--index-list", type=Path, default=None)
    ap.add_argument("--null-list", type=Path, default=None)
    ap.add_argument("--index-pdb-dir", type=Path, default=None)
    ap.add_argument("--null-pdb-dir", type=Path, default=None)
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output-dir", type=Path, default=Path("validation_scripts/output"))
    ap.add_argument("--tmalign", type=str, default="TMalign")
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    index_list = (args.index_list or (repo_root / "04-folddisco-validation-preparation/data/domain-list-index.txt")).resolve()
    null_list = (args.null_list or (repo_root / "04-folddisco-validation-preparation/data/domain-list-noncluster.txt")).resolve()
    index_pdb_dir = (args.index_pdb_dir or (repo_root / "04-folddisco-validation-preparation/data/index_pdbs")).resolve()
    null_pdb_dir = (args.null_pdb_dir or (repo_root / "04-folddisco-validation-preparation/data/non_cluster_pdbs")).resolve()
    out_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [p for p in [index_list, null_list, index_pdb_dir, null_pdb_dir] if not p.exists()]
    if missing:
        print("[WARN] Missing required inputs:")
        for p in missing:
            print(f"  - {p}")
        print("[WARN] Provide local data paths via --index-pdb-dir/--null-pdb-dir and rerun.")

    rows_index = read_domain_list(index_list) if index_list.exists() else []
    rows_null = read_domain_list(null_list) if null_list.exists() else []

    existing_index_ids = {p.stem for p in index_pdb_dir.glob("*.pdb")} if index_pdb_dir.exists() else set()
    existing_null_ids = {p.stem for p in null_pdb_dir.glob("*.pdb")} if null_pdb_dir.exists() else set()

    rng = random.Random(args.seed)
    true_pairs = build_true_pairs(rows_index, existing_index_ids, args.sample_size, rng)
    false_pairs = build_false_pairs(rows_index, rows_null, existing_index_ids, existing_null_ids, args.sample_size, rng)

    tmalign_bin = shutil.which(args.tmalign)
    use_tmalign = tmalign_bin is not None

    rows_out = []
    true_tm, false_tm = [], []
    true_high, false_low = 0, 0

    def eval_pair(set_name: str, pair: Pair) -> None:
        nonlocal true_high, false_low
        id_a, id_b = pair
        pdb_a = (null_pdb_dir / f"{id_a}.pdb") if set_name == "false" and (null_pdb_dir / f"{id_a}.pdb").exists() else (index_pdb_dir / f"{id_a}.pdb")
        pdb_b = (index_pdb_dir / f"{id_b}.pdb")
        if not pdb_a.exists() or not pdb_b.exists():
            return

        metrics = run_tmalign(pdb_a, pdb_b, tmalign_bin) if use_tmalign and tmalign_bin else None
        if metrics is None:
            metrics = fallback_pair_metrics(pdb_a, pdb_b, id_a, id_b)
        if metrics is None:
            return

        tm, rmsd, method = metrics
        rows_out.append({
            "set": set_name,
            "id_a": id_a,
            "id_b": id_b,
            "tm_score": f"{tm:.5f}",
            "rmsd": f"{rmsd:.5f}",
            "method": method,
        })

        if set_name == "true":
            true_tm.append(tm)
            if tm > 0.5:
                true_high += 1
        else:
            false_tm.append(tm)
            if tm < 0.5:
                false_low += 1

    for pair in true_pairs:
        eval_pair("true", pair)
    for pair in false_pairs:
        eval_pair("false", pair)

    metrics_tsv = out_dir / "tm_rmsd_sampled_pairs.tsv"
    with metrics_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["set", "id_a", "id_b", "tm_score", "rmsd", "method"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_out)

    plot_warning = plot_tm_distribution(out_dir / "tm_score_distribution.png", true_tm, false_tm)

    summary_txt = out_dir / "validate_sets_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as handle:
        handle.write("Validation Task 1: Structural Integrity of True/False Sets\n")
        handle.write(f"index_list={index_list}\n")
        handle.write(f"null_list={null_list}\n")
        handle.write(f"index_pdb_dir={index_pdb_dir}\n")
        handle.write(f"null_pdb_dir={null_pdb_dir}\n")
        handle.write(f"sample_size_requested={args.sample_size}\n")
        handle.write(f"true_pairs_sampled={len(true_pairs)}\n")
        handle.write(f"false_pairs_sampled={len(false_pairs)}\n")
        handle.write(f"true_pairs_scored={len(true_tm)}\n")
        handle.write(f"false_pairs_scored={len(false_tm)}\n")
        handle.write(f"true_tm_mean={(np.mean(true_tm) if true_tm else float('nan')):.5f}\n")
        handle.write(f"false_tm_mean={(np.mean(false_tm) if false_tm else float('nan')):.5f}\n")
        handle.write(f"true_fraction_tm_gt_0_5={(true_high / len(true_tm) if true_tm else float('nan')):.5f}\n")
        handle.write(f"false_fraction_tm_lt_0_5={(false_low / len(false_tm) if false_tm else float('nan')):.5f}\n")
        handle.write(f"scoring_method={'TMalign' if use_tmalign else 'fallback_kabsch'}\n")
        if plot_warning:
            handle.write(f"plot_warning={plot_warning}\n")

    print(f"[INFO] Wrote {metrics_tsv}")
    print(f"[INFO] Wrote {summary_txt}")
    print(f"[INFO] Wrote {out_dir / 'tm_score_distribution.png'}")


if __name__ == "__main__":
    main()
