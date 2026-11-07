# CUDA-aware MPI microbenchmark

`cuda_aware_mpi_bench.cu` measures what CUDA-aware MPI buys a GPU solver.
Each test runs in two modes:

- **device**: GPU pointers go straight to `MPI_Send`/`MPI_Recv`/`MPI_Isend`.
  UCX moves the data over NVLink (`cuda_ipc`) inside a node and over
  InfiniBand between nodes.
- **staged**: data is copied to pinned host memory, sent, and copied back.
  This is what an application has to do with an MPI that is not CUDA-aware.

There are two tests:

- `pingpong`: two ranks exchange one contiguous message from 8 B to 256 MiB.
- `halo`: each rank owns an `N^3` block of doubles in a 3-D periodic process
  grid. It packs its six faces with GPU kernels, exchanges them with its six
  neighbours, and unpacks them. This is the ghost-cell fill pattern of a
  stencil or fluid solver. Per-rank work is fixed, so this is weak scaling.
  The reported time is the slowest rank's mean per exchange.

Every received buffer is checked on the GPU. Each output row carries a
`verified` flag, and all published rows have `verified=1`.

## Build

```bash
nvcc -O3 -std=c++17 -arch=sm_90 -I$MPI_HOME/include cuda_aware_mpi_bench.cu \
     -o cuda_aware_mpi_bench -L$MPI_HOME/lib -lmpi
```

`MPI_HOME` must point to a CUDA-aware MPI. The published runs used
Open MPI 4.1.6 with UCX 1.17 on NVIDIA H100 SXM GPUs (8 per node, NVLink),
CUDA 12.6.

## Run

```bash
# one node, NVLink
mpirun -np 2 --mca pml ucx -x UCX_TLS=sm,self,cuda_copy,cuda_ipc \
       ./cuda_aware_mpi_bench pingpong nvlink_same_node > pingpong.csv
mpirun -np 8 --mca pml ucx -x UCX_TLS=sm,self,cuda_copy,cuda_ipc \
       ./cuda_aware_mpi_bench halo one_node_8 > halo_8.csv
# two nodes, InfiniBand
mpirun -np 2 --map-by ppr:1:node --mca pml ucx \
       -x UCX_TLS=rc,sm,self,cuda_copy,cuda_ipc -x UCX_IB_GPU_DIRECT_RDMA=no \
       ./cuda_aware_mpi_bench pingpong infiniband_two_nodes > pingpong_ib.csv
```

Each rank selects its GPU from `OMPI_COMM_WORLD_LOCAL_RANK` (or
`PMI_LOCAL_RANK`).

## Notes from the measured system

- **GPUDirect RDMA was not available.** The InfiniBand adapters refused to
  register GPU memory (dmabuf registration failed). With
  `UCX_IB_GPU_DIRECT_RDMA=no`, UCX still accepts device pointers but pipelines
  them through host memory internally.
- **The UCX UD transport timed out on this fabric**, so transports were pinned
  with `UCX_TLS` as shown above. One node consistently failed the inter-node
  test and was excluded.
- The previous multi-GPU configuration used Open MPI's `ob1` PML over TCP and
  shared-memory BTLs. That configuration is measured too, as the baseline.

The results are in
[`data/benchmarks/cuda_aware_mpi_*.csv`](../../data/benchmarks/README.md) and
discussed in
[GPU scaling and capacity, section 4](../../docs/gpu-scaling-and-capacity.md#4-cuda-aware-mpi).
