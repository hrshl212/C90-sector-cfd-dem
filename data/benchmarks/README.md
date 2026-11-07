# GPU benchmark tables

Aggregate timings from completed one-node NVIDIA H100 runs. Each table backs a
section of [GPU capacity, solver acceleration, and multi-GPU scaling](../../docs/gpu-scaling-and-capacity.md).
`scripts/plot_gpu_benchmarks.py` regenerates the figures from these files.

## `single_gpu_particle_capacity.csv`

Wall time per CFD step on one H100 versus particle count and neighbor-search
configuration. Empty cells were not separately reported. `fluid_s` is the
midpoint of the reported per-step range. The `crashed_32bit_csr_overflow`
row has no timing because the step did not finish.

## `multi_gpu_strong_scaling.csv`

Mean per-step stage times over 100 steps for a 45° sector with 55,954
particles on 1, 2, 4, and 8 H100s. Stage timers can overlap, so `total_s` is
measured directly and is not the sum of the stage columns.

## `pressure_solver_ab_tests.csv`

Matched control/candidate times (seconds) for the C90 pressure-projection
optimizations. Rows are sequential A/B tests with different baselines, so
their gains must not be multiplied together.
