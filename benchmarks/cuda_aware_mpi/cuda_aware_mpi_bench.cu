// CUDA-aware MPI versus host-staged MPI on NVIDIA GPUs.
//
// Two tests:
//   pingpong : ranks 0 and 1 exchange one contiguous message, 8 B .. 256 MiB.
//   halo     : every rank owns an N^3 double block in a 3-D periodic process
//              grid, packs its six faces on the GPU, exchanges them with its
//              neighbours, and unpacks them (the communication pattern of a
//              stencil / fluid solver ghost-cell fill).
// Each test runs in two modes:
//   device : GPU pointers are passed directly to MPI (CUDA-aware path; UCX
//            moves data over NVLink (cuda_ipc) within a node and over InfiniBand
//            between nodes).
//   staged : data is copied to pinned host memory, sent with MPI, and copied
//            back (what a non-CUDA-aware MPI forces the application to do).
// Each received buffer is checked for correctness before timing is reported.

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                 \
  do {                                                                        \
    cudaError_t e_ = (x);                                                     \
    if (e_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_),      \
                   __FILE__, __LINE__);                                       \
      MPI_Abort(MPI_COMM_WORLD, 1);                                           \
    }                                                                         \
  } while (0)

static int g_rank = 0, g_size = 1;

__global__ void fill(char* p, size_t n, int seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    p[i] = (char)((i * 131u + seed) & 0x7f);
}

__global__ void count_bad(const char* p, size_t n, int seed,
                          unsigned long long* bad) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    if (p[i] != (char)((i * 131u + seed) & 0x7f)) atomicAdd(bad, 1ull);
}

static bool verify(const char* d, size_t n, int seed) {
  unsigned long long* bad;
  CK(cudaMalloc(&bad, sizeof(*bad)));
  CK(cudaMemset(bad, 0, sizeof(*bad)));
  count_bad<<<1024, 256>>>(d, n, seed, bad);
  unsigned long long h = 0;
  CK(cudaMemcpy(&h, bad, sizeof(h), cudaMemcpyDeviceToHost));
  CK(cudaFree(bad));
  return h == 0;
}

// ---------------------------------------------------------------- ping-pong
static void pingpong(bool device, size_t bytes, int iters, double* lat_us,
                     double* bw_gbs, bool* ok) {
  char *dsend, *drecv, *hsend = nullptr, *hrecv = nullptr;
  CK(cudaMalloc(&dsend, bytes));
  CK(cudaMalloc(&drecv, bytes));
  if (!device) {
    CK(cudaMallocHost(&hsend, bytes));
    CK(cudaMallocHost(&hrecv, bytes));
  }
  fill<<<1024, 256>>>(dsend, bytes, 7);
  CK(cudaMemset(drecv, 0, bytes));
  CK(cudaDeviceSynchronize());
  int peer = 1 - g_rank;
  int n = (int)bytes;

  auto one = [&](bool sender) {
    if (sender) {
      if (device) {
        MPI_Send(dsend, n, MPI_CHAR, peer, 0, MPI_COMM_WORLD);
      } else {
        CK(cudaMemcpy(hsend, dsend, bytes, cudaMemcpyDeviceToHost));
        MPI_Send(hsend, n, MPI_CHAR, peer, 0, MPI_COMM_WORLD);
      }
    } else {
      if (device) {
        MPI_Recv(drecv, n, MPI_CHAR, peer, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(hrecv, n, MPI_CHAR, peer, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        CK(cudaMemcpy(drecv, hrecv, bytes, cudaMemcpyHostToDevice));
      }
    }
  };

  int warm = std::max(2, iters / 10);
  double t0 = 0;
  for (int it = 0; it < warm + iters; ++it) {
    if (it == warm) {
      MPI_Barrier(MPI_COMM_WORLD);
      t0 = MPI_Wtime();
    }
    if (g_rank == 0) {
      one(true);
      one(false);
    } else {
      one(false);
      one(true);
    }
  }
  double dt = (MPI_Wtime() - t0) / iters / 2.0;  // one-way time
  bool good = verify(drecv, bytes, 7);
  int g = good, gall = 0;
  MPI_Allreduce(&g, &gall, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  *lat_us = dt * 1e6;
  *bw_gbs = bytes / dt / 1e9;
  *ok = gall;
  CK(cudaFree(dsend));
  CK(cudaFree(drecv));
  if (!device) {
    CK(cudaFreeHost(hsend));
    CK(cudaFreeHost(hrecv));
  }
}

// --------------------------------------------------------------------- halo
// Face packing for an (N+2)^3 block with one ghost layer.
__device__ __forceinline__ size_t idx3(int i, int j, int k, int M) {
  return ((size_t)k * M + j) * M + i;
}

// face f: 0=-x 1=+x 2=-y 3=+y 4=-z 5=+z ; interior cells 1..N
__global__ void pack_face(const double* u, double* buf, int N, int f) {
  int M = N + 2;
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y * blockDim.y + threadIdx.y;
  if (a >= N || b >= N) return;
  int p = (f & 1) ? N : 1;
  size_t s;
  if (f < 2) s = idx3(p, a + 1, b + 1, M);
  else if (f < 4) s = idx3(a + 1, p, b + 1, M);
  else s = idx3(a + 1, b + 1, p, M);
  buf[(size_t)b * N + a] = u[s];
}

__global__ void unpack_face(double* u, const double* buf, int N, int f) {
  int M = N + 2;
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y * blockDim.y + threadIdx.y;
  if (a >= N || b >= N) return;
  int p = (f & 1) ? N + 1 : 0;  // ghost layer
  size_t s;
  if (f < 2) s = idx3(p, a + 1, b + 1, M);
  else if (f < 4) s = idx3(a + 1, p, b + 1, M);
  else s = idx3(a + 1, b + 1, p, M);
  u[s] = buf[(size_t)b * N + a];
}

__global__ void init_block(double* u, int N, int rank) {
  int M = N + 2;
  size_t total = (size_t)M * M * M;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total;
       i += (size_t)gridDim.x * blockDim.x)
    u[i] = rank * 1.0e6 + (double)(i % 1000003);
}

static void halo(bool device, int N, int iters, double* t_ms, bool* ok) {
  int dims[3] = {0, 0, 0}, periods[3] = {1, 1, 1}, coords[3];
  MPI_Dims_create(g_size, 3, dims);
  MPI_Comm cart;
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);
  MPI_Cart_coords(cart, g_rank, 3, coords);
  int nb[6];
  for (int d = 0; d < 3; ++d) MPI_Cart_shift(cart, d, 1, &nb[2 * d], &nb[2 * d + 1]);

  int M = N + 2;
  size_t face = (size_t)N * N;
  double* u;
  CK(cudaMalloc(&u, sizeof(double) * M * M * M));
  init_block<<<1024, 256>>>(u, N, g_rank);
  std::vector<double*> ds(6), dr(6), hs(6, nullptr), hr(6, nullptr);
  for (int f = 0; f < 6; ++f) {
    CK(cudaMalloc(&ds[f], face * sizeof(double)));
    CK(cudaMalloc(&dr[f], face * sizeof(double)));
    if (!device) {
      CK(cudaMallocHost(&hs[f], face * sizeof(double)));
      CK(cudaMallocHost(&hr[f], face * sizeof(double)));
    }
  }
  dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
  int cnt = (int)face;

  auto exchange = [&]() {
    for (int f = 0; f < 6; ++f) pack_face<<<grd, blk>>>(u, ds[f], N, f);
    CK(cudaDeviceSynchronize());
    MPI_Request req[12];
    for (int f = 0; f < 6; ++f) {
      // data for my face f goes to neighbour nb[f], which stores it in its
      // opposite ghost face (f ^ 1).
      double* rb = device ? dr[f] : hr[f];
      MPI_Irecv(rb, cnt, MPI_DOUBLE, nb[f], f ^ 1, cart, &req[f]);
    }
    for (int f = 0; f < 6; ++f) {
      double* sb = ds[f];
      if (!device) {
        CK(cudaMemcpy(hs[f], ds[f], face * sizeof(double), cudaMemcpyDeviceToHost));
        sb = hs[f];
      }
      MPI_Isend(sb, cnt, MPI_DOUBLE, nb[f], f, cart, &req[6 + f]);
    }
    MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
    for (int f = 0; f < 6; ++f) {
      if (!device)
        CK(cudaMemcpy(dr[f], hr[f], face * sizeof(double), cudaMemcpyHostToDevice));
      unpack_face<<<grd, blk>>>(u, dr[f], N, f);
    }
    CK(cudaDeviceSynchronize());
  };

  for (int it = 0; it < 3; ++it) exchange();
  MPI_Barrier(cart);
  double t0 = MPI_Wtime();
  for (int it = 0; it < iters; ++it) exchange();
  double dt = (MPI_Wtime() - t0) / iters;
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MAX, cart);

  // Correctness: ghost face f must equal neighbour nb[f]'s interior face f^1.
  // Neighbour data is rank*1e6 + index pattern; compare via a reference copy.
  bool good = true;
  {
    std::vector<double> ref(face), got(face);
    for (int f = 0; f < 6 && good; ++f) {
      // Rebuild the neighbour's packed face on the host.
      int src = nb[f], sf = f ^ 1, p = (sf & 1) ? N : 1;
      for (int b = 0; b < N; ++b)
        for (int a = 0; a < N; ++a) {
          size_t s;
          if (sf < 2) s = ((size_t)(b + 1) * M + (a + 1)) * M + p;
          else if (sf < 4) s = ((size_t)(b + 1) * M + p) * M + (a + 1);
          else s = ((size_t)p * M + (b + 1)) * M + (a + 1);
          ref[(size_t)b * N + a] = src * 1.0e6 + (double)(s % 1000003);
        }
      CK(cudaMemcpy(got.data(), dr[f], face * sizeof(double), cudaMemcpyDeviceToHost));
      good = std::memcmp(ref.data(), got.data(), face * sizeof(double)) == 0;
    }
  }
  int g = good, gall = 0;
  MPI_Allreduce(&g, &gall, 1, MPI_INT, MPI_MIN, cart);
  *t_ms = dt * 1e3;
  *ok = gall;
  for (int f = 0; f < 6; ++f) {
    CK(cudaFree(ds[f]));
    CK(cudaFree(dr[f]));
    if (!device) {
      CK(cudaFreeHost(hs[f]));
      CK(cudaFreeHost(hr[f]));
    }
  }
  CK(cudaFree(u));
  MPI_Comm_free(&cart);
}

int main(int argc, char** argv) {
  // Select the GPU before MPI_Init so UCX registers the right device.
  const char* lr = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!lr) lr = std::getenv("PMI_LOCAL_RANK");
  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  CK(cudaSetDevice(lr ? std::atoi(lr) % ndev : 0));
  CK(cudaFree(0));

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);
  std::string test = argc > 1 ? argv[1] : "pingpong";
  std::string label = argc > 2 ? argv[2] : "run";

  if (test == "pingpong") {
    if (g_size != 2) {
      if (!g_rank) std::fprintf(stderr, "pingpong needs 2 ranks\n");
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (!g_rank) std::printf("test,label,mode,bytes,latency_us,bandwidth_GBs,verified\n");
    for (size_t b = 8; b <= (size_t)256 << 20; b *= 4) {
      int iters = b <= (1 << 16) ? 2000 : (b <= (16 << 20) ? 100 : 20);
      for (int m = 0; m < 2; ++m) {
        double lat, bw;
        bool ok;
        pingpong(m == 0, b, iters, &lat, &bw, &ok);
        if (!g_rank)
          std::printf("pingpong,%s,%s,%zu,%.3f,%.3f,%d\n", label.c_str(),
                      m == 0 ? "device" : "staged", b, lat, bw, (int)ok);
      }
    }
  } else if (test == "halo") {
    if (!g_rank) std::printf("test,label,mode,ranks,block_N,face_bytes,exchange_ms,verified\n");
    for (int N : {64, 128, 256, 384}) {
      for (int m = 0; m < 2; ++m) {
        double t;
        bool ok;
        halo(m == 0, N, N >= 256 ? 20 : 100, &t, &ok);
        if (!g_rank)
          std::printf("halo,%s,%s,%d,%d,%zu,%.4f,%d\n", label.c_str(),
                      m == 0 ? "device" : "staged", g_size, N,
                      (size_t)N * N * sizeof(double), t, (int)ok);
      }
    }
  }
  std::fflush(stdout);
  MPI_Finalize();
  return 0;
}
