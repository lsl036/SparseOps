# Figure 7 Artifact Evaluation

This experiment sweeps the mixed-accumulator L2 budget from `0.1` to `1.1`
and records L2-related traffic for `gupta3`. It uses the saved constrained-LSH
ordering, so clustering preprocessing is not repeated.

## 1. Install prerequisites

On a Fedora, RHEL, or openEuler system:

```bash
sudo dnf install -y gcc gcc-c++ make cmake python3 python3-pip curl tar zstd
python3 -m pip install --user ssgetpy requests tqdm matplotlib openpyxl
```

LIKWID 5.5.1 includes its required Lua and hwloc sources.

## 2. Prepare SparseOps inputs

Run these commands from the SparseOps repository root:

```bash
cd /path/to/SparseOps

python3 data/download_datasets.py --source test --skip-existing
python3 data/download_datasets.py --reordering --skip-existing

test -f data/gupta3/gupta3.mtx
test -f data/reordering/lsh_order/gupta3.perm
test -f data/reordering/lsh_order/gupta3.offsets
```

The reference files used for the reported AMD run have these checksums:

```text
3dbee39916cf747d4fc42bed720774231f30e635972ef915c93867e21ec8bc95  data/gupta3/gupta3.mtx
aaf1bbf6ba8e4ae20c4a0859a385e6d5e7c2a7af4499772052f023e3ce7f2526  data/reordering/lsh_order/gupta3.perm
ddd1ca9e3cffebbe74c1637c1831b136c318c9ff26a3ca4a1fd41006ded4d2fb  data/reordering/lsh_order/gupta3.offsets
```

Verify them with:

```bash
sha256sum \
  data/gupta3/gupta3.mtx \
  data/reordering/lsh_order/gupta3.perm \
  data/reordering/lsh_order/gupta3.offsets
```

## 3. Build SparseOps

Configure explicitly with GCC for reproducibility, then build only the required
test program and its library dependency:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++

cmake --build build -j "$(nproc)" --target test_spgemm_lsh
test -x build/test_spgemm_lsh
```

The reference AMD run used GCC 10.3.1, CMake 3.22.0, and Linux
5.10.0-60.18.0.50.oe2203.x86_64.

## 4. Download LIKWID 5.5.1

Skip the download and extraction if `figs/fig7/likwid-5.5.1` is included in
the artifact:

```bash
cd figs/fig7

curl -fLO https://ftp.fau.de/pub/likwid/likwid-5.5.1.tar.gz
tar -xzf likwid-5.5.1.tar.gz
test -d likwid-5.5.1
```

## 5. Build local LIKWID with perf_event

The local build avoids the root-owned MSR access daemon. `ACCESSMODE` must be
passed to both build commands because it is a compile-time setting.

```bash
cd likwid-5.5.1

make distclean
make -j "$(nproc)" ACCESSMODE=perf_event
make ACCESSMODE=perf_event local

LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ./likwid-perfctr -v

cd ..
```

Do not run `make install` for this rootless workflow. The experiment script
automatically sets the local library path and generates a temporary LIKWID
configuration that points to the local performance-group files.

## 6. Check perf_event permissions

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

The AMD branch uses process-scoped custom PMC events and was verified with
`perf_event_paranoid=2`. The Intel `-g L3` group may require a value of `1` or
lower. If LIKWID reports `Permission denied`, run:

```bash
sudo sysctl -w kernel.perf_event_paranoid=1
```

## 7. Run the sweep

From `figs/fig7`:

```bash
./run_l2_volume.sh
```

The default run performs one warmup and 10 measured SpGEMM iterations for each
of the 11 L2 fractions. The script sets `OMP_PLACES=cores` and
`OMP_PROC_BIND=spread` unless they are already set.

Platform-specific collection:

- Intel uses CPUs `0-27`, 28 SpGEMM threads, and LIKWID group `L3`. It prefers
  `L3|MEM evict data volume [GBytes]` and falls back to
  `L3 evict data volume [GBytes]` when required by the microarchitecture.
- AMD selects one hardware thread from every physical core. On the reference
  EPYC 7C13 this is CPUs `0-127` and 128 SpGEMM threads. It measures
  `L2_PF_HIT_IN_L3`, `L2_PF_MISS_IN_L3`, and
  `L2_CACHE_MISS_AFTER_L1_MISS`, then multiplies their per-thread sum by
  64 bytes to obtain L2 fill/miss traffic in GB.

To override affinity or the number of numeric iterations:

```bash
CPU_LIST=0-27 THREADS=28 SPGEMM_ITERATIONS=10 ./run_l2_volume.sh
```

## 8. Validate the result

The output is `l2_volume.txt`. It has one header and 11 data rows:

```text
l2_fraction SUM Min Max Avg
0.1 ...
...
1.1 ...
```

Validate its shape and fraction range:

```bash
awk '
  NR == 1 && $0 != "l2_fraction SUM Min Max Avg" { exit 1 }
  NR > 1 && (NF != 5 || $1 != sprintf("%.1f", (NR - 1) / 10)) { exit 1 }
  END { if (NR != 12) exit 1 }
' l2_volume.txt

wc -l l2_volume.txt
```

The result is written atomically only after all 11 configurations succeed.
`Sum`, `Min`, `Max`, and `Avg` are statistics across the measured hardware
threads, expressed in GBytes.

## 9. Generate the paper plot

The plotting script reads the `L2_hitrate` worksheet from the Figure 6 Excel
workbook and does not depend on `l2_volume.txt`:

```bash
python3 plot_l2_budget.py
test -f L2_budget.pdf
```

The default paths are:

```text
input:  ../fig6/hyper_params.xlsx
output: L2_budget.pdf
```

Both paths can be overridden explicitly:

```bash
python3 plot_l2_budget.py \
  --input ../fig6/hyper_params.xlsx \
  --sheet L2_hitrate \
  --output L2_budget.pdf
```
