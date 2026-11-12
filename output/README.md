# Output index

Canonical deliverables:

- `image_demo/dashboard.png`
  - calibrated raw-image pipeline card for all three target types.
- `image_demo/report.json`
  - measured dimensions, predicted uncertainty, timing and diagnostics.
- `benchmark_combined/summary.json`
  - three independent seeds, 252 repeated image measurements.
- `benchmark_combined/image_benchmark_combined.csv`
  - every case and every repeat.
- `parallax_benchmark/parallax_summary.json`
  - top-plane height compensation before/after comparison.
- `parallax_benchmark/parallax_benchmark.csv`
  - individual parallax cases.

Other directories are development checkpoints and smoke tests. The canonical
results above are the ones used in `README.md` and
`docs/05_competitive_upgrade_and_validation.md`.
