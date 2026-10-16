# Pageable pool performance checks

Build with `BUILD_BENCHMARKS=ON`, then run
`target/jni/cmake-build/benchmarks/PAGEABLE_POOL_BENCH`. Add
`target/jni/cmake-build` and `target/jni/cmake-build/lib` to
`LD_LIBRARY_PATH`. Use `--list` to confirm the benchmark names and axes.

Run each comparison on an otherwise idle host three times, save CSV with
`--csv <file>`, and compare the median **CPU Time (sec)**. The initialization
and allocator paths are host work; their reported GPU timer is not the useful
metric. For DtoH, report copy time and effective bandwidth. Record the GPU,
CPU, NUMA topology, kernel, driver, pool size, and benchmark commit with each
result. Keep allocation size, thread count, and THP policy fixed when changing
one technique.

| Question | Benchmark and axes | Baseline and candidate |
| --- | --- | --- |
| Pre-touch method | `PageablePool Init`, `size=1073741824`, `pretouch_threads=8` | `method=manual`, `populate`, `memset` |
| Transparent huge pages | `PageablePool DtoH Copy`, `size=268435456`, `numa=default` | `thp=default`, `deny` |
| GPU-local placement | `PageablePool DtoH Copy`, `thp=default` | `numa=default`, `gpu_local` |
| Contended allocator | `PageablePool AllocFree *`, `chunk=65536`, `threads=16` | list, indexed, arena |
| Fragmentation | `PageablePool Fragmented *` | list, indexed |
| Cached split tail | `PageablePool Remainder Cache Burst` | `cache=off`, `on` |

For example:

```bash
build_dir=target/jni/cmake-build
LD_LIBRARY_PATH="$build_dir:$build_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$build_dir/benchmarks/PAGEABLE_POOL_BENCH" \
  --benchmark 'PageablePool Init' --axis size=1073741824 \
  --axis pretouch_threads=8 --csv pageable-init.csv
```

On a single-node host, or when the GPU's PCI sysfs `numa_node` is `-1`,
`numa=gpu_local` falls back to the default policy; use a multi-node host for
the placement comparison. The arena divides capacity into fixed shards, so
benchmark gains for small allocations do not imply it can satisfy the same
large allocations as the unsharded pool.

## Local check: RTX 5880 Ada, October 13, 2026

Host: Threadripper PRO 5975WX, one NUMA node, Linux 6.17, NVIDIA driver
590.48.01. The GPU PCI sysfs `numa_node` is `-1`, and THP is set to
`[madvise]`. Initialization, allocator, fragmentation, and cache results are
ranges from two runs. The THP and DtoH-reference checks were run once.

| Comparison | CPU time | Observation |
| --- | --- | --- |
| 1 GiB init, 8 workers: populate / manual / memset | 63–64 / 73–74 / 459–461 ms | Populate is ~14% faster than manual; memset is ~7× slower. |
| 64 KiB alloc/free, 16 workers: list / indexed / arena | 473–485 / 1060–1126 / 104–106 µs per 64 operations per worker | Arena is ~4.5× faster than list; indexed is slower without fragmentation. |
| Fragmented list / indexed | 27.1–27.4 / 12.1–12.3 µs per allocation/free | Indexed is ~2.2× faster under ~2048 separated holes. |
| Remainder cache burst: off / on | 11.66–11.69 / 11.66–11.75 µs per 32-allocation burst | No measurable gain. |

At 1 GiB, THP deny was about 1% faster than default (67.9 versus 68.7 ms),
which is too small to establish a benefit here. With THP set to `madvise`,
the default pool backing is unlikely to be promoted to huge pages. GPU-local
NUMA placement cannot be evaluated on this host. At 256 MiB, a pooled DtoH
copy took 16.8 ms versus 143 ms into fresh malloc memory and 10.2 ms into
pinned memory.

## H100 NVL check, October 13, 2026

Host: two AMD EPYC 9354 CPUs, two NUMA nodes, Linux 6.5, NVIDIA driver
580.173.02. The H100 is on NUMA node 1. THP is set to `[madvise]`. The exact
benchmark executable and `libcudf.so` matching source commit `4c169504` were
copied from the local worktree;
`CUDA_VISIBLE_DEVICES=1` selected the H100. The host's CUDA toolkit is 12.6;
the copied binary was built with CUDA 12.9. Each configuration ran three times,
with 15 NVBench samples per run; the table reports the median of the three
reported times. Raw results are in
[the H100 CSV](pageable_pool_h100_2026-10-13.csv).

| Comparison | Median time | Observation |
| --- | --- | --- |
| 1 GiB init, 8 workers: populate / manual / memset | 51.9 / 78.7 / 492.1 ms CPU | Populate is 34% faster than manual; memset is 9.5x slower than populate. |
| 64 KiB alloc/free, 16 workers: list / indexed / arena | 942 / 1904 / 201 µs CPU per 64 operations per worker | Arena is 4.7x faster than list; indexed loses without fragmentation. |
| Fragmented list / indexed | 29.3 / 12.8 µs CPU per allocation/free | Indexed is 2.3x faster with separated holes. |
| Remainder cache burst: off / on | 12.6 / 13.2 µs CPU per burst | No gain. |
| 256 MiB DtoH: default / THP deny | 11.835 / 11.892 ms GPU | Difference is below 1%. |
| 256 MiB DtoH: default / GPU-local NUMA | 11.835 / 11.754 ms GPU | Difference is below 1%. `mbind(MPOL_PREFERRED, node 1)` succeeded in a separate trace. |
| 256 MiB DtoH: pool / fresh malloc / pinned | 11.835 / 137.592 / 5.029 ms GPU | Pre-touched pool is much faster than fresh pageable memory. |

The GPU-local NUMA and THP comparisons were much smaller than the observed
run-to-run variation and do not establish a benefit.

## EC2 g6.4xlarge check: L4, October 13, 2026

Host: 16 vCPUs on AMD EPYC 7R13, one NUMA node, Ubuntu 22.04, Linux 6.8,
NVIDIA driver 595.91.07, one NVIDIA L4. THP is set to `[madvise]`. The same
benchmark executable and `libcudf.so` matching source commit `4c169504` were used
for the H100 check and this check. Each configuration ran three times, with 15
NVBench samples per run; the table reports the median of the three reported
times. Raw results are in [the L4 CSV](pageable_pool_l4_g6_2026-10-13.csv).
GPU-local NUMA placement was omitted because this instance has one NUMA node.

| Comparison | Median time | Observation |
| --- | --- | --- |
| 1 GiB init, 8 workers: populate / manual / memset | 58.9 / 73.4 / 471.8 ms CPU | Populate is 20% faster than manual; memset is 8.0x slower than populate. |
| 64 KiB alloc/free, 16 workers: list / indexed / arena | 410 / 1019 / 123 µs CPU per 64 operations per worker | Arena is 3.3x faster than list; indexed loses without fragmentation. |
| Fragmented list / indexed | 33.5 / 14.9 µs CPU per allocation/free | Indexed is 2.2x faster with separated holes. |
| Remainder cache burst: off / on | 14.5 / 14.2 µs CPU per burst | No convincing gain. |
| 256 MiB DtoH: default / THP deny | 20.697 / 20.696 ms GPU | Effectively identical. |
| 256 MiB DtoH: pool / fresh malloc / pinned | 20.697 / 155.882 / 20.319 ms GPU | Pre-touched pool is much faster than fresh pageable memory. |
