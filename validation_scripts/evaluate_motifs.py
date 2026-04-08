#!/usr/bin/env python3
"""Evaluate motif plausibility using 3D geometric features (no secondary-structure labels)."""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("[ERROR] numpy is required for evaluate_motifs.py") from exc

try:
    from Bio.PDB import PDBParser
except Exception as exc:  # pragma: no cover
    raise SystemExit("[ERROR] biopython is required for evaluate_motifs.py") from exc


ResidueTag = Tuple[str, int, str]  # chain, resseq, icode


def parse_tag(tag: str) -> Optional[ResidueTag]:
    tag = tag.strip()
    m = re.match(r"^([A-Za-z0-9])(\d+)([A-Za-z]?)$", tag)
    if not m:
        return None
    chain = m.group(1)
    resseq = int(m.group(2))
    icode = m.group(3) or " "
    return chain, resseq, icode


def parse_chain_residue_list(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for i, raw in enumerate(handle):
            if i == 0 and raw.lower().startswith("cat_id"):
                continue
            parts = raw.strip().split("\t")
            if len(parts) < 3:
                continue
            cat_id, pdb_id, tags_raw = parts[0], parts[1], parts[2]
            tags = []
            for t in tags_raw.split(","):
                parsed = parse_tag(t)
                if parsed:
                    tags.append(parsed)
            if not tags:
                continue
            rows.append({"cat_id": cat_id, "pdb_id": pdb_id, "tags": tags})
    return rows


def get_ca_coord(structure, chain: str, resseq: int, icode: str):
    model = structure[0]
    if chain not in model:
        return None
    chain_obj = model[chain]
    key = (" ", resseq, icode)
    alt_key = (" ", resseq, " ")
    residue = chain_obj[key] if key in chain_obj else (chain_obj[alt_key] if alt_key in chain_obj else None)
    if residue is None or "CA" not in residue:
        return None
    return residue["CA"].coord


def kabsch_align(ref: np.ndarray, mob: np.ndarray) -> Tuple[np.ndarray, float]:
    ref_center = ref.mean(axis=0)
    mob_center = mob.mean(axis=0)

    ref_centered = ref - ref_center
    mob_centered = mob - mob_center

    covariance = mob_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(covariance)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T

    mob_aligned = (mob_centered @ rot) + ref_center
    diffs = ref - mob_aligned
    rmsd = float(np.sqrt((diffs * diffs).sum() / len(ref)))
    return mob_aligned, rmsd


def tm_like_score_from_distances(distances: np.ndarray, length_norm: int) -> float:
    normalized_length = max(length_norm, 1)
    if normalized_length <= 15:
        d0 = 0.5
    else:
        d0 = 1.24 * ((normalized_length - 15) ** (1.0 / 3.0)) - 1.8
        d0 = max(d0, 0.5)
    return float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))


def motif_pair_geometry(coords_a: np.ndarray, coords_b: np.ndarray) -> Optional[Tuple[float, float]]:
    if len(coords_a) != len(coords_b) or len(coords_a) < 3:
        return None
    aligned_b, rmsd = kabsch_align(coords_a, coords_b)
    dists = np.linalg.norm(coords_a - aligned_b, axis=1)
    tm_like = tm_like_score_from_distances(dists, len(coords_a))
    return rmsd, tm_like


def radius_of_gyration(coords: np.ndarray) -> float:
    center = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))


def max_pairwise_distance(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return float("nan")
    diffs = coords[:, None, :] - coords[None, :, :]
    dmat = np.linalg.norm(diffs, axis=2)
    return float(np.max(dmat))


def plot_geometry_distribution(out_png: Path, pair_rmsd: List[float], pair_tm: List[float]) -> Optional[str]:
    """Plot motif pair geometry distributions and return optional warning string."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib is not installed; plot generation skipped"

    if not pair_rmsd or not pair_tm:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No pairwise motif geometry data available", ha="center", va="center", fontsize=11)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Metric")
        plt.ylabel("Density")
        plt.title("Motif Pairwise Geometry Distribution")
        plt.tight_layout()
        plt.savefig(out_png, dpi=220)
        plt.close()
        return "no pairwise motif geometry data available"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(pair_rmsd, bins=30, density=True, alpha=0.75, color="#1f77b4")
    axes[0].set_xlabel("Pairwise motif RMSD (Å)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("RMSD")

    axes[1].hist(pair_tm, bins=np.linspace(0.0, 1.0, 31), density=True, alpha=0.75, color="#2ca02c")
    axes[1].axvline(0.5, linestyle="--", color="black", linewidth=1.0)
    axes[1].set_xlabel("Pairwise motif TM-like score")
    axes[1].set_ylabel("Density")
    axes[1].set_title("TM-like")

    fig.suptitle("Motif Pairwise Geometry Distribution")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return None


def main() -> None:
    repo_root_default = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description="Evaluate motif spatial plausibility using 3D geometric metrics")
    ap.add_argument("--repo-root", type=Path, default=repo_root_default)
    ap.add_argument("--chain-residue-list", type=Path, default=None)
    ap.add_argument("--index-pdb-dir", type=Path, default=None)
    ap.add_argument("--max-motifs", type=int, default=500)
    ap.add_argument("--continuity-threshold", type=float, default=4.5, help="Consecutive CA distance threshold in Angstrom")
    ap.add_argument("--pairwise-rmsd-threshold", type=float, default=3.0, help="RMSD threshold for geometric motif coherence")
    ap.add_argument("--pairwise-tm-threshold", type=float, default=0.5, help="TM-like threshold for geometric motif coherence")
    ap.add_argument("--output-dir", type=Path, default=Path("validation_scripts/output"))
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    chain_residue_list = (args.chain_residue_list or (repo_root / "04-folddisco-validation-preparation/result/chain_residue_list.txt")).resolve()
    index_pdb_dir = (args.index_pdb_dir or (repo_root / "04-folddisco-validation-preparation/data/index_pdbs")).resolve()
    out_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chain_residue_list.exists():
        print(f"[WARN] Missing chain_residue_list: {chain_residue_list}")
        return
    if not index_pdb_dir.exists():
        print(f"[WARN] Missing index_pdb_dir: {index_pdb_dir}")
        print("[WARN] Provide local PDB path via --index-pdb-dir and rerun.")

    entries = parse_chain_residue_list(chain_residue_list)[: args.max_motifs]

    parser = PDBParser(QUIET=True)

    max_consecutive_dists: List[float] = []
    rg_values: List[float] = []
    max_pairwise_values: List[float] = []
    continuity_ok = 0
    continuity_total = 0

    per_motif_rows: List[Dict[str, str]] = []
    motifs_by_cat: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)

    for ent in entries:
        cat_id = str(ent["cat_id"])
        pdb_id = str(ent["pdb_id"])
        tags: List[ResidueTag] = ent["tags"]  # type: ignore[assignment]

        pdb_path = index_pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            continue

        try:
            structure = parser.get_structure(pdb_id, str(pdb_path))
        except Exception:
            continue

        coords = []
        for chain, resseq, icode in tags:
            coord = get_ca_coord(structure, chain, resseq, icode)
            if coord is None:
                continue
            coords.append(coord)

        if len(coords) < 2:
            continue

        coords_arr = np.array(coords, dtype=float)
        dists = np.linalg.norm(coords_arr[1:] - coords_arr[:-1], axis=1)
        max_consecutive = float(np.max(dists)) if len(dists) else float("nan")
        rg = radius_of_gyration(coords_arr)
        max_pairwise = max_pairwise_distance(coords_arr)

        continuity_total += 1
        continuity_flag = max_consecutive <= args.continuity_threshold
        if continuity_flag:
            continuity_ok += 1

        max_consecutive_dists.append(max_consecutive)
        rg_values.append(rg)
        max_pairwise_values.append(max_pairwise)

        motifs_by_cat[cat_id].append((pdb_id, coords_arr))

        per_motif_rows.append(
            {
                "cat_id": cat_id,
                "pdb_id": pdb_id,
                "motif_length": str(len(coords_arr)),
                "max_consecutive_ca_distance": f"{max_consecutive:.5f}",
                "radius_of_gyration": f"{rg:.5f}",
                "max_pairwise_ca_distance": f"{max_pairwise:.5f}",
                "continuity_pass": str(continuity_flag),
            }
        )

    pair_rows: List[Dict[str, str]] = []
    pair_rmsd_values: List[float] = []
    pair_tm_values: List[float] = []
    pair_pass = 0

    for cat_id, motif_list in motifs_by_cat.items():
        for (pdb_a, coords_a), (pdb_b, coords_b) in itertools.combinations(motif_list, 2):
            metrics = motif_pair_geometry(coords_a, coords_b)
            if metrics is None:
                continue
            rmsd, tm_like = metrics
            rmsd_pass = rmsd <= args.pairwise_rmsd_threshold
            tm_pass = tm_like >= args.pairwise_tm_threshold
            joint_pass = rmsd_pass and tm_pass
            if joint_pass:
                pair_pass += 1

            pair_rmsd_values.append(rmsd)
            pair_tm_values.append(tm_like)
            pair_rows.append(
                {
                    "cat_id": cat_id,
                    "pdb_id_a": pdb_a,
                    "pdb_id_b": pdb_b,
                    "motif_length": str(len(coords_a)),
                    "pair_rmsd": f"{rmsd:.5f}",
                    "pair_tm_like": f"{tm_like:.5f}",
                    "rmsd_pass": str(rmsd_pass),
                    "tm_pass": str(tm_pass),
                    "joint_pass": str(joint_pass),
                }
            )

    per_motif_tsv = out_dir / "motif_spatial_quality.tsv"
    with per_motif_tsv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "cat_id",
            "pdb_id",
            "motif_length",
            "max_consecutive_ca_distance",
            "radius_of_gyration",
            "max_pairwise_ca_distance",
            "continuity_pass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(per_motif_rows)

    pair_tsv = out_dir / "motif_pairwise_geometry.tsv"
    with pair_tsv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "cat_id",
            "pdb_id_a",
            "pdb_id_b",
            "motif_length",
            "pair_rmsd",
            "pair_tm_like",
            "rmsd_pass",
            "tm_pass",
            "joint_pass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(pair_rows)

    plot_warning = plot_geometry_distribution(out_dir / "motif_geometry_distribution.png", pair_rmsd_values, pair_tm_values)

    summary = out_dir / "evaluate_motifs_summary.txt"
    with summary.open("w", encoding="utf-8") as handle:
        handle.write("Validation Task 3: Motif 3D Geometric Plausibility\n")
        handle.write(f"chain_residue_list={chain_residue_list}\n")
        handle.write(f"index_pdb_dir={index_pdb_dir}\n")
        handle.write(f"entries_read={len(entries)}\n")
        handle.write(f"entries_scored={len(per_motif_rows)}\n")
        handle.write(f"continuity_threshold={args.continuity_threshold}\n")
        handle.write(f"pairwise_rmsd_threshold={args.pairwise_rmsd_threshold}\n")
        handle.write(f"pairwise_tm_threshold={args.pairwise_tm_threshold}\n")
        handle.write(f"continuity_pass_fraction={(continuity_ok / continuity_total if continuity_total else float('nan')):.5f}\n")
        handle.write(f"mean_max_consecutive_ca_distance={(mean(max_consecutive_dists) if max_consecutive_dists else float('nan')):.5f}\n")
        handle.write(f"mean_radius_of_gyration={(mean(rg_values) if rg_values else float('nan')):.5f}\n")
        handle.write(f"mean_max_pairwise_ca_distance={(mean(max_pairwise_values) if max_pairwise_values else float('nan')):.5f}\n")
        handle.write(f"pairwise_comparisons={len(pair_rows)}\n")
        handle.write(f"mean_pair_rmsd={(mean(pair_rmsd_values) if pair_rmsd_values else float('nan')):.5f}\n")
        handle.write(f"mean_pair_tm_like={(mean(pair_tm_values) if pair_tm_values else float('nan')):.5f}\n")
        handle.write(f"pairwise_joint_pass_fraction={(pair_pass / len(pair_rows) if pair_rows else float('nan')):.5f}\n")
        if plot_warning:
            handle.write(f"plot_warning={plot_warning}\n")

    print(f"[INFO] Wrote {per_motif_tsv}")
    print(f"[INFO] Wrote {pair_tsv}")
    print(f"[INFO] Wrote {summary}")
    print(f"[INFO] Wrote {out_dir / 'motif_geometry_distribution.png'}")


if __name__ == "__main__":
    main()
