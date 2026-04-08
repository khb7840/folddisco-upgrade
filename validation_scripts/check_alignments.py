#!/usr/bin/env python3
"""Validate Foldmason alignment quality using gap/entropy and dominant-residue metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def read_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    current = None
    buf: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    seqs[current] = "".join(buf)
                current = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line)
    if current is not None:
        seqs[current] = "".join(buf)
    return seqs


def col_entropy(chars: List[str]) -> float:
    counts: Dict[str, int] = {}
    total = 0
    for c in chars:
        if c == "-":
            continue
        counts[c] = counts.get(c, 0) + 1
        total += 1
    if total == 0:
        return 0.0
    ent = 0.0
    for n in counts.values():
        p = n / total
        ent -= p * math.log2(p)
    return ent


def normalize_char(c: str) -> str:
    if c == "-" or c.islower():
        return "-"
    return c.upper()


def analyze_alignment(seqs: Dict[str, str], dominant_cutoff: float) -> Dict[str, float]:
    if not seqs:
        return {}

    lengths = {len(s) for s in seqs.values()}
    if len(lengths) != 1:
        return {}

    n_seq = len(seqs)
    aln_len = next(iter(lengths))
    matrix = [[normalize_char(c) for c in s] for s in seqs.values()]

    gap_fracs: List[float] = []
    entropies: List[float] = []
    dominant_fracs: List[float] = []
    dominant_pass_strict = 0

    for j in range(aln_len):
        col = [row[j] for row in matrix]
        gap_count = sum(1 for x in col if x == "-")
        gap_frac = gap_count / n_seq
        gap_fracs.append(gap_frac)

        non_gap = [x for x in col if x != "-"]
        if non_gap:
            counts: Dict[str, int] = {}
            for x in non_gap:
                counts[x] = counts.get(x, 0) + 1
            max_count = max(counts.values())
            dominant = max_count / n_seq
        else:
            dominant = 0.0
        dominant_fracs.append(dominant)

        if gap_count == 0 and dominant >= dominant_cutoff:
            dominant_pass_strict += 1

        entropies.append(col_entropy(col))

    return {
        "num_sequences": float(n_seq),
        "alignment_length": float(aln_len),
        "mean_gap_fraction": mean(gap_fracs),
        "columns_gap_gt_50pct": float(sum(1 for x in gap_fracs if x > 0.5)),
        "columns_gap_gt_50pct_frac": float(sum(1 for x in gap_fracs if x > 0.5) / aln_len),
        "mean_entropy": mean(entropies),
        "mean_dominant_fraction": mean(dominant_fracs),
        "dominant_pass_strict_count": float(dominant_pass_strict),
        "dominant_pass_strict_frac": float(dominant_pass_strict / aln_len),
    }


def main() -> None:
    repo_root_default = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description="Check Foldmason alignment quality")
    ap.add_argument("--repo-root", type=Path, default=repo_root_default)
    ap.add_argument("--foldmason-dir", type=Path, default=None, help="Directory with per-group folders containing result_aa.fa")
    ap.add_argument("--fasta-name", type=str, default="result_aa.fa")
    ap.add_argument("--dominant-cutoff", type=float, default=0.66)
    ap.add_argument("--output-dir", type=Path, default=Path("validation_scripts/output"))
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    foldmason_dir = (args.foldmason_dir or (repo_root / "04-folddisco-validation-preparation/data/foldmason_1")).resolve()
    out_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_paths = sorted(foldmason_dir.glob(f"*/{args.fasta_name}")) if foldmason_dir.exists() else []
    if not fasta_paths:
        print(f"[WARN] No alignment files found under: {foldmason_dir}")
        print("[WARN] Provide local Foldmason extraction path via --foldmason-dir and rerun.")

    rows: List[Dict[str, str]] = []
    agg = {
        "n_alignments": 0,
        "n_failed_format": 0,
        "mean_gap_fraction": [],
        "mean_entropy": [],
        "dominant_pass_strict_frac": [],
        "columns_gap_gt_50pct_frac": [],
    }

    for fasta in fasta_paths:
        seqs = read_fasta(fasta)
        stats = analyze_alignment(seqs, args.dominant_cutoff)
        if not stats:
            agg["n_failed_format"] += 1
            continue

        agg["n_alignments"] += 1
        agg["mean_gap_fraction"].append(stats["mean_gap_fraction"])
        agg["mean_entropy"].append(stats["mean_entropy"])
        agg["dominant_pass_strict_frac"].append(stats["dominant_pass_strict_frac"])
        agg["columns_gap_gt_50pct_frac"].append(stats["columns_gap_gt_50pct_frac"])

        rows.append(
            {
                "group_id": fasta.parent.name,
                "num_sequences": str(int(stats["num_sequences"])),
                "alignment_length": str(int(stats["alignment_length"])),
                "mean_gap_fraction": f"{stats['mean_gap_fraction']:.5f}",
                "columns_gap_gt_50pct": str(int(stats["columns_gap_gt_50pct"])),
                "columns_gap_gt_50pct_frac": f"{stats['columns_gap_gt_50pct_frac']:.5f}",
                "mean_entropy": f"{stats['mean_entropy']:.5f}",
                "mean_dominant_fraction": f"{stats['mean_dominant_fraction']:.5f}",
                "dominant_pass_strict_count": str(int(stats["dominant_pass_strict_count"])),
                "dominant_pass_strict_frac": f"{stats['dominant_pass_strict_frac']:.5f}",
            }
        )

    per_alignment_tsv = out_dir / "alignment_quality_per_group.tsv"
    with per_alignment_tsv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "group_id",
            "num_sequences",
            "alignment_length",
            "mean_gap_fraction",
            "columns_gap_gt_50pct",
            "columns_gap_gt_50pct_frac",
            "mean_entropy",
            "mean_dominant_fraction",
            "dominant_pass_strict_count",
            "dominant_pass_strict_frac",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary = out_dir / "check_alignments_summary.txt"
    with summary.open("w", encoding="utf-8") as handle:
        handle.write("Validation Task 2: Foldmason Alignment Quality\n")
        handle.write(f"foldmason_dir={foldmason_dir}\n")
        handle.write(f"fasta_name={args.fasta_name}\n")
        handle.write(f"dominant_cutoff={args.dominant_cutoff}\n")
        handle.write(f"alignments_found={len(fasta_paths)}\n")
        handle.write(f"alignments_analyzed={agg['n_alignments']}\n")
        handle.write(f"alignments_failed_format={agg['n_failed_format']}\n")

        if agg["n_alignments"] > 0:
            handle.write(f"global_mean_gap_fraction={mean(agg['mean_gap_fraction']):.5f}\n")
            handle.write(f"global_mean_entropy={mean(agg['mean_entropy']):.5f}\n")
            handle.write(f"global_mean_dominant_pass_strict_frac={mean(agg['dominant_pass_strict_frac']):.5f}\n")
            handle.write(f"global_mean_columns_gap_gt_50pct_frac={mean(agg['columns_gap_gt_50pct_frac']):.5f}\n")

            risky = sum(1 for v in agg["columns_gap_gt_50pct_frac"] if v > 0.5)
            handle.write(f"alignments_with_majority_gappy_columns={risky}\n")

    print(f"[INFO] Wrote {per_alignment_tsv}")
    print(f"[INFO] Wrote {summary}")


if __name__ == "__main__":
    main()
