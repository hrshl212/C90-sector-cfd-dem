# GPU benchmark tables

Aggregate timings from completed NVIDIA H100 runs. Each table backs a
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

## `cuda_aware_mpi_pingpong.csv`

Ping-pong latency and bandwidth between two GPUs from
[`benchmarks/cuda_aware_mpi`](../../benchmarks/cuda_aware_mpi/README.md).
`link` is `nvlink_same_node`, `same_node`, `infiniband_two_nodes`, or
`two_nodes`. `mpi_transport` is `pml_ucx` (CUDA-aware) or the previous
`pml_ob1_tcp*` configuration. `buffer=device` passes GPU pointers to MPI;
`buffer=staged` copies through pinned host memory first. Bandwidth for tiny
messages rounds to zero. For `ob1`, `device` means the GPU pointer is handed
to Open MPI's internal CUDA staging.

## `cuda_aware_mpi_halo.csv`

Mean six-face halo-exchange time (slowest rank) with an `block_N^3` double
block per GPU, including GPU pack and unpack. `face_bytes` is the size of one
face message.
