# Validation Report

## Scope
This report summarizes integrity checks for the Folddisco validation inputs and generated motif quality.

- Task 1: Structural integrity of True/False sets (TM-score/RMSD)
- Task 2: Foldmason alignment quality (gap and entropy statistics)
- Task 3: Motif 3D geometric plausibility

## Execution Context
- Repository: `khb7840/folddisco-upgrade`
- Script directory: `validation_scripts/`
- Output directory (default): `validation_scripts/output/`

## Scripts Added
- `validation_scripts/validate_sets.py`
- `validation_scripts/check_alignments.py`
- `validation_scripts/evaluate_motifs.py`

## How to Run
From repository root:

```bash
python validation_scripts/validate_sets.py \
  --index-pdb-dir /absolute/path/to/data/index_pdbs \
  --null-pdb-dir /absolute/path/to/data/non_cluster_pdbs

python validation_scripts/check_alignments.py \
  --foldmason-dir /absolute/path/to/foldmason_output_root

python validation_scripts/evaluate_motifs.py \
  --index-pdb-dir /absolute/path/to/data/index_pdbs
```

## Output Files
Expected generated outputs in `validation_scripts/output/`:

- `tm_rmsd_sampled_pairs.tsv`
- `validate_sets_summary.txt`
- `tm_score_distribution.png`
- `alignment_quality_per_group.tsv`
- `check_alignments_summary.txt`
- `motif_spatial_quality.tsv`
- `motif_pairwise_geometry.tsv`
- `evaluate_motifs_summary.txt`
- `motif_geometry_distribution.png`

## Current Run Summary (this environment)
The repository clone does not include local structural PDB datasets under:

- `04-folddisco-validation-preparation/data/index_pdbs/`
- `04-folddisco-validation-preparation/data/non_cluster_pdbs/`
- Foldmason MSA directories with `result_aa.fa`

As a result, execution completes in warning mode and reports missing inputs instead of biological conclusions.

Observed run outputs:

- `true_pairs_scored=0`, `false_pairs_scored=0`
- `alignments_found=0`, `alignments_analyzed=0`
- `entries_read=500`, `entries_scored=0`
- Placeholder visualizations were still generated:
  - `validation_scripts/output/tm_score_distribution.png`
  - `validation_scripts/output/motif_geometry_distribution.png`

Repository test status after script changes:

- `python 05-folddisco-validation/command/09_value_check_test.py` → **PASS**

## Interpretation Guidance
- True-set TM-score distribution should be shifted toward high similarity (`TM-score > 0.5`).
- False-set TM-score distribution should represent structural null/background (`TM-score < 0.5`).
- High `columns_gap_gt_50pct_frac` or low `dominant_pass_strict_frac` can indicate poor alignment blocks.
- Motifs with high consecutive Cα jumps, high intra-motif spread, high pairwise RMSD, or low pairwise TM-like scores may indicate spatially incoherent query extraction.

## Warnings / Edge Cases
- If TM-align binary is unavailable, Task 1 falls back to Kabsch-based approximate TM-score.
- If matplotlib is unavailable, PNG plots are skipped and warnings are written in summaries.
- Task 3 now avoids secondary-structure labels and uses geometry-only metrics:
  - per-motif continuity (`max_consecutive_ca_distance`)
  - per-motif compactness (`radius_of_gyration`, `max_pairwise_ca_distance`)
  - within-group pairwise motif similarity (`pair_rmsd`, `pair_tm_like`)
