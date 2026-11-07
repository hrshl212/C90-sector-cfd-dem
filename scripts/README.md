# Public scripts

`plot_particle_population.py` is an independent plotting utility that consumes
only the sanitized CSV files in this repository. It writes normalized
retention and raw-count figures.

```bash
python3 scripts/plot_particle_population.py \
  --retention-csv data/sanitized/particle_retention_timeseries.csv \
  --count-csv data/sanitized/particle_count_timeseries.csv \
  --output-dir build/figures
```

`check_release.py` validates required files, CSV invariants, local Markdown
links, and the absence of common private-workspace identifiers.

`render_snapshot_comparison.py` reads matched AMReX field and particle
plotfiles and produces the longitudinal-flow, transverse cross-section, and
particle snapshot figures shown in the main README. It
requires NumPy, Matplotlib, and a VTK build that includes the AMReX
grid and particle readers. Raw simulation plotfiles are intentionally outside
this repository.

```bash
python3 scripts/render_snapshot_comparison.py \
  --full-flow /path/to/full/plotfile \
  --sector-flow /path/to/sector/field-plotfile \
  --sector-particles /path/to/sector/particle-plotfile \
  --output-dir build/snapshots
```

Use `--full-particles` only when the full-cylinder particles are stored in a
different plotfile directory from its flow fields. The piston overlay defaults
to `z0 = 0.2 mm`, speed `1.7 mm/s`, and radius `4.33 mm`; the corresponding
`--piston-z0`, `--piston-speed`, and `--piston-radius` options use SI units.

`plot_gpu_benchmarks.py` reads the tables in `data/benchmarks/` and writes the
single-GPU capacity and multi-GPU strong-scaling figures used in
[`docs/gpu-scaling-and-capacity.md`](../docs/gpu-scaling-and-capacity.md).

```bash
python3 scripts/plot_gpu_benchmarks.py --output-dir build/figures
```
