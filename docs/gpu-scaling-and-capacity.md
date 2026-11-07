# GPU capacity, solver acceleration, and multi-GPU scaling

This page extends [GPU performance engineering](gpu-performance.md) with three
measured campaigns from the same rotational-sector CFD-DEM framework:

1. pushing one NVIDIA H100 from 20 million to 180 million particles;
2. accelerating the rotational pressure projection with a GPU sparse-operator cache,
   synchronization removal, kernel fusion, and a better Krylov bottom solver; and
3. measuring multi-GPU strong scaling and explaining why it did not pay off.

Every number below comes from a completed run and a matched control. The raw
tables are in [`data/benchmarks/`](../data/benchmarks/README.md), and
[`scripts/plot_gpu_benchmarks.py`](../scripts/plot_gpu_benchmarks.py)
regenerates the figures.

## 1. 180 million particles on one H100

The particle-capacity study used a 180° sector of the same injector, with
polydisperse 6/12/40 µm particle mixtures at 20M and 80M particles (30/55/15%
by mass). The goal was to
learn how far one 80 GB H100 could go before multi-GPU decomposition became
necessary.

![Single-H100 particle capacity](../figures/performance/single_gpu_particle_capacity.png)

**20M particles: profile first, then remove the dominant cost.** A one-step
stage profile assigned 15.30 s of the 16.55 s DEM time (92.5%) to rebuilding
the neighbor list on every DEM substep. Contact kernels took only 1.22 s. The
fix rebuilds the list once per CFD step and relies on its geometric skin
between rebuilds. Rebuild intervals of 2, 4, 8, and 16 substeps were each
validated in turn:

| Neighbor-list rebuild interval | Sustained s/step | Speedup | Max. pressure change |
|---|---:|---:|---:|
| every substep (baseline) | 17.8 | 1.0× | — |
| 2 | 10.1 | 1.8× | 0.048% |
| 4 | 6.35 | 2.8× | 0.055% |
| 8 | 4.40 | 4.0× | 0.048% |
| 16 (once per CFD step) | 3.49 | **5.1×** | 0.055% |

At interval 2, force and velocity were bitwise identical after one step.

**80M particles: diagnosing a 32-bit overflow.** At 80,039,438 particles the
solver crashed with a CUDA illegal-memory-access error. A 64-bit audit of
neighbor counts, taken before the compressed-sparse-row (CSR) scan, found
**31,222,780,504** candidate entries. That is 7.27× the `2^32 − 1` limit of the
32-bit CSR offsets, and the index list alone would have needed about 125 GB.
So the crash was a wrapped write while filling the list, not a fault in the
contact-force kernel. Widening the offsets to 64 bits could not have fixed it,
because the list would still not fit in 80 GB.

The repair was algorithmic. A size-aware neighbor search sorts particles into
three diameter bins with refinement ratios 6, 3, and 1. Small–small pairs then
no longer search out to the 40 µm radius of the largest particles.

| Metric | Uniform search | Size-aware bins | Change |
|---|---:|---:|---:|
| Neighbor candidate entries | 31.2 billion | 153.4 million | **203.5× fewer** |
| Maximum list length per particle | 492 | 4 | — |
| Result | crash (overflow) | stable, 3.70 s/step | runs |

A permanent guard now aborts with a clear message if the count ever exceeds
the 32-bit limit, instead of writing past the end of the list.

Single-variable controls tuned the remaining settings:

- A coarser `5 3 1` bin layout was rejected because it was 23% slower
  (4.54 s/step).
- Cutting DEM substeps from 16 to 4 sped the step up by 8.4% (3.39 s/step). In
  a matched 30-step check, velocity stayed within 0.001% and piston pressure
  within 0.136%.
- A coarser particle-force update interval was rejected. It saved only 0.02
  s/step and shifted pressure by 2.7–4.5%.

**180M particles: memory-bound capacity.** A 4.5/10/40 µm timing cloud
with 180,000,037 particles, using size-aware bins with refinement ratios 8, 4,
and 1, ran stably at **6.01 s/step** on one H100. Of that, DEM took 4.49 s,
of which the neighbor-list rebuild was 4.07 s. Fluid took 1.27 s and coupling
0.26 s. Peak device memory was 63.6 GB of 81.1 GB. At 180M the remaining
bottleneck is the neighbor-list rebuild. This is a capacity and timing result
only: its particles were laid out in slabs, not as a mixed production cloud.

A next step would be to reuse the neighbor list across CFD steps. A design
review rejected that as unsafe for this geometry. About 2.0 million rotating
seam-image particles continuously enter and leave the image band. Keeping their
list indices stale breaks correctness, and refreshing them invalidates the
list. Doing this properly needs a new cyclic-aware Verlet list with a stable
mapping from each source particle to its image, not a small patch. That
decision is documented here rather than shipped as an unsafe option.

## 2. Accelerating the rotational pressure projection (C90)

In the C90 sector, the pressure Poisson equation is solved on the rotational
quotient space `PᵀAP`. The coarse levels of the multigrid hierarchy apply a
recursive Galerkin operator `R PᵀAP I`. A nested GPU timer breakdown over about
48,000 recursive calls found where the time went:

| Recursive coarse-operator stage | Share of time |
|---|---:|
| Interpolation (prolongation) | 42.7% |
| Restriction | 27.9% |
| Fine-level operator apply | 9.9% |
| Fine-level `P` reconstruction | 2.9% |
| Buffer clear | 1.5% |

Each change below was tested as a matched on/off A/B on the same hardware. A
change was accepted only if step-1 force was unchanged and convergence was
preserved:

| Optimization | Control | Candidate | Gain |
|---|---:|---:|---:|
| Remove device-to-host syncs in the one-rank, one-box hierarchy | 13.88 s | 9.27 s | **33.2%** |
| Fuse interpolation with output initialization | 7.99 s | 7.60 s | 4.9% |
| Fuse the fine-level `P` reconstruction in place | 7.60 s | 7.26 s | 4.5% |
| Assemble the recursive Galerkin operator once as a width-48 ELLPACK sparse matrix on the GPU | 7.26 s | 5.07 s | **30.1%** |

The ELLPACK cache stores 921,370 exact entries, with a maximum row width of 42.
Its action matches the recursive operator to `6.2e-15`, with the cached matrix
built in 3.5 s. A graph audit first measured the operator's true sparsity, which
showed that a dense response cache would need 783 components. It also rejected
a cheaper 3-color cache, which was wrong by `1.6` because the coupling extends
beyond the assumed block.

Rejected candidates included a parameter cache that was 1.8% slower, a fused
restriction/quotient transfer that was exact but did not speed up the step,
and removing a buffer clear, which stalls the solve because embedded-boundary
cells retain stale values.

**A better bottom solver.** The coarsest-level BiCGStab solve was replaced by a
variant that keeps its best iterate and allows an early exit only after at
least a 10× residual reduction. Over 90 coupled steps (11–100), this cut
summed nodal-solve time from 1822.7 s to 895.5 s (**2.04×**) and wall time
from 31:31 to 16:16 (**1.94×**). Step-100 pressure stayed within `0.015%` of
the full-cylinder reference.

**Net result.** With these solver paths, the accurate C90 sector runs steps
21–100 in 254.6 s versus 325.8 s for the full cylinder. That is **1.28× higher
throughput**, with a mean pressure error of `0.110%` against the full cylinder.

## 3. Multi-GPU strong scaling

The two-rank, two-GPU correctness tests in
[validation results](validation-results.md) show that the rotational seam is
correct across MPI ranks. In particular, cross-rank contact symmetry holds to
`2.78e-15` and one- versus two-rank piston force agrees to `3.96e-9`. In a CPU audit, a
distributed gather path for the cyclic pressure operator converged to identical
global norms on one and two ranks.

Strong scaling was then measured on a 45° sector with 55,954 particles, using
1, 2, 4, and 8 H100s for 100 steps each:

![Multi-GPU strong scaling](../figures/performance/multi_gpu_strong_scaling.png)

| GPUs | Fluid | Coupling | DEM | Total s/step | vs. 1 GPU |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.239 | 0.126 | 0.019 | **0.387** | 1.00× |
| 2 | 0.380 | 0.215 | 0.019 | 0.617 | 0.63× |
| 4 | 0.710 | 0.311 | 0.016 | 1.038 | 0.37× |
| 8 | 0.523 | 0.222 | 0.011 | 0.761 | 0.51× |

**Diagnosis.**

- The DEM stage does strong-scale, reaching 1.8× on 8 GPUs.
- The mesh has only five fluid boxes, so eight ranks cannot all receive fluid
  work.
- The fluid solve, the rotational pressure coupling, and the particle-to-fluid
  exchange become communication-bound. Together they cost more than one GPU's
  entire step.

The production decision was therefore to stay on one H100 and spend effort on
the single-GPU optimizations above. A multi-GPU speedup would first require:

- a decomposition with many more fluid boxes than ranks;
- a communication path for the rotational coupling that avoids gathering to
  one rank; and
- a sustained multi-rank validation of the coupled pressure solve, which is
  listed as unqualified in [limitations](limitations.md).

No GPU-aware MPI timing is claimed here. Those runs were not part of the
qualified campaign.

## Engineering practices shown

- **Profile before optimizing.** Stage timers, nested GPU timers, Nsight
  Compute, and graph-sparsity audits chose every target.
- **Validate each change against a matched control.** Each A/B checked force,
  pressure, particle state, and particle count, not just wall time.
- **Fix the cause of a crash, not the symptom.** A 32-bit CSR overflow was
  traced to an algorithmic candidate explosion and fixed by size-aware binning.
- **Accept the accuracy budget explicitly.** Faster but less accurate
  configurations were rejected with documented error numbers.
- **Report negative results.** Multi-GPU slowdowns, slower fusions, and unsafe
  list reuse are documented alongside the wins.
