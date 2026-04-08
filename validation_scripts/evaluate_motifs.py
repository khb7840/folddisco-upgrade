#!/usr/bin/env python3
"""Evaluate spatial and biological plausibility of generated motifs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
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


Tag = Tuple[str, int, str]  # chain, resseq, icode


def parse_tag(tag: str) -> Optional[Tag]:
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


def build_secondary_index_from_header(pdb_path: Path) -> Dict[Tuple[str, int, str], str]:
    sec: Dict[Tuple[str, int, str], str] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            rec = line[:6].strip()
            if rec == "HELIX":
                chain = line[19:20].strip() or " "
                start = int(line[21:25].strip())
                icode_start = (line[25:26].strip() or " ")
                end = int(line[33:37].strip())
                icode_end = (line[37:38].strip() or " ")
                if chain == " ":
                    continue
                for resseq in range(start, end + 1):
                    sec[(chain, resseq, " ")] = "H"
                    sec[(chain, resseq, icode_start)] = "H"
                    sec[(chain, resseq, icode_end)] = "H"
            elif rec == "SHEET":
                chain = line[21:22].strip() or " "
                start = int(line[22:26].strip())
                icode_start = (line[26:27].strip() or " ")
                end = int(line[33:37].strip())
                icode_end = (line[37:38].strip() or " ")
                if chain == " ":
                    continue
                for resseq in range(start, end + 1):
                    sec[(chain, resseq, " ")] = "E"
                    sec[(chain, resseq, icode_start)] = "E"
                    sec[(chain, resseq, icode_end)] = "E"
    return sec


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


def plot_secondary_structure_distribution(out_png: Path, counts: Counter) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib is not installed; plot generation skipped"

    labels = ["H", "E", "C"]
    values = [counts.get("H", 0), counts.get("E", 0), counts.get("C", 0)]
    if sum(values) == 0:
        return "no secondary structure assignments available"

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    plt.xlabel("Secondary Structure Class")
    plt.ylabel("Residue Count")
    plt.title("Motif Secondary Structure Distribution")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    return None


def main() -> None:
    repo_root_default = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description="Evaluate motif spatial continuity and structure plausibility")
    ap.add_argument("--repo-root", type=Path, default=repo_root_default)
    ap.add_argument("--chain-residue-list", type=Path, default=None)
    ap.add_argument("--index-pdb-dir", type=Path, default=None)
    ap.add_argument("--max-motifs", type=int, default=500)
    ap.add_argument("--continuity-threshold", type=float, default=4.5, help="Consecutive CA distance threshold in Angstrom")
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
    sec_counts: Counter = Counter()
    max_dists: List[float] = []
    continuity_ok = 0
    continuity_total = 0
    rows_out: List[Dict[str, str]] = []

    for ent in entries:
        pdb_id = ent["pdb_id"]
        tags: List[Tag] = ent["tags"]  # type: ignore[assignment]
        pdb_path = index_pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            continue

        try:
            structure = parser.get_structure(pdb_id, str(pdb_path))
        except Exception:
            continue

        sec_index = build_secondary_index_from_header(pdb_path)

        coords = []
        motif_sec = []
        for chain, resseq, icode in tags:
            coord = get_ca_coord(structure, chain, resseq, icode)
            if coord is None:
                continue
            coords.append(coord)
            motif_sec.append(sec_index.get((chain, resseq, icode), sec_index.get((chain, resseq, " "), "C")))

        if len(coords) < 2:
            continue

        coords_arr = np.array(coords, dtype=float)
        dists = np.linalg.norm(coords_arr[1:] - coords_arr[:-1], axis=1)
        max_dist = float(np.max(dists))
        max_dists.append(max_dist)

        continuity_total += 1
        continuity_flag = max_dist <= args.continuity_threshold
        if continuity_flag:
            continuity_ok += 1

        for ss in motif_sec:
            if ss not in {"H", "E"}:
                ss = "C"
            sec_counts[ss] += 1

        rows_out.append(
            {
                "cat_id": str(ent["cat_id"]),
                "pdb_id": str(pdb_id),
                "motif_length": str(len(coords)),
                "max_consecutive_ca_distance": f"{max_dist:.5f}",
                "continuity_pass": str(continuity_flag),
                "helix_residues": str(sum(1 for x in motif_sec if x == "H")),
                "sheet_residues": str(sum(1 for x in motif_sec if x == "E")),
                "coil_residues": str(sum(1 for x in motif_sec if x not in {"H", "E"})),
            }
        )

    per_motif_tsv = out_dir / "motif_spatial_quality.tsv"
    with per_motif_tsv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "cat_id",
            "pdb_id",
            "motif_length",
            "max_consecutive_ca_distance",
            "continuity_pass",
            "helix_residues",
            "sheet_residues",
            "coil_residues",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_out)

    plot_warning = plot_secondary_structure_distribution(out_dir / "motif_secondary_structure_distribution.png", sec_counts)

    total_ss = sum(sec_counts.values())
    summary = out_dir / "evaluate_motifs_summary.txt"
    with summary.open("w", encoding="utf-8") as handle:
        handle.write("Validation Task 3: Motif Spatial and Biological Plausibility\n")
        handle.write(f"chain_residue_list={chain_residue_list}\n")
        handle.write(f"index_pdb_dir={index_pdb_dir}\n")
        handle.write(f"entries_read={len(entries)}\n")
        handle.write(f"entries_scored={len(rows_out)}\n")
        handle.write(f"continuity_threshold={args.continuity_threshold}\n")
        handle.write(f"continuity_pass_fraction={(continuity_ok / continuity_total if continuity_total else float('nan')):.5f}\n")
        handle.write(f"mean_max_consecutive_ca_distance={(mean(max_dists) if max_dists else float('nan')):.5f}\n")
        handle.write(f"helix_fraction={(sec_counts.get('H', 0) / total_ss if total_ss else float('nan')):.5f}\n")
        handle.write(f"sheet_fraction={(sec_counts.get('E', 0) / total_ss if total_ss else float('nan')):.5f}\n")
        handle.write(f"coil_fraction={(sec_counts.get('C', 0) / total_ss if total_ss else float('nan')):.5f}\n")
        if plot_warning:
            handle.write(f"plot_warning={plot_warning}\n")

    print(f"[INFO] Wrote {per_motif_tsv}")
    print(f"[INFO] Wrote {summary}")
    print(f"[INFO] Wrote {out_dir / 'motif_secondary_structure_distribution.png'}")


if __name__ == "__main__":
    main()
