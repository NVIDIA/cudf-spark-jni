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
| Parallel copies | `PageablePool DtoH Chunked`, `size=1073741824` | `chunks=1`, `2`, `4`, `8` |

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
ranges from two runs. The chunked-copy, THP, and DtoH-reference checks were
run once.

| Comparison | CPU time | Observation |
| --- | --- | --- |
| 1 GiB init, 8 workers: populate / manual / memset | 63–64 / 73–74 / 459–461 ms | Populate is ~14% faster than manual; memset is ~7× slower. |
| 64 KiB alloc/free, 16 workers: list / indexed / arena | 473–485 / 1060–1126 / 104–106 µs per 64 operations per worker | Arena is ~4.5× faster than list; indexed is slower without fragmentation. |
| Fragmented list / indexed | 27.1–27.4 / 12.1–12.3 µs per allocation/free | Indexed is ~2.2× faster under ~2048 separated holes. |
| Remainder cache burst: off / on | 11.66–11.69 / 11.66–11.75 µs per 32-allocation burst | No measurable gain. |
| 1 GiB DtoH: 1 / 2 / 4 / 8 chunks | 68.4 / 67.8 / 68.6 / 68.3 ms | No useful gain from splitting the copy. |

At 1 GiB, THP deny was about 1% faster than default (67.9 versus 68.7 ms),
which is too small to establish a benefit here. With THP set to `madvise`,
the default pool backing is unlikely to be promoted to huge pages. GPU-local
NUMA placement cannot be evaluated on this host. At 256 MiB, a pooled DtoH
copy took 16.8 ms versus 143 ms into fresh malloc memory and 10.2 ms into
pinned memory.
