# Fig9 - 20 Cases Analysis

## Run 20 representative matrices performance test

```bash
cd $SPARSEOPS_ROOT/figs/fig9/

export OMP_PLACES=cores
export OMP_PROC_BIND=spread

python3 $SPARSEOPS_ROOT/script/run_test_spgemm_hc_lsh_list.py $SPARSEOPS_ROOT/runable_casesets.txt --base-dir $SPARSEOPS_ROOT/data --kernel 3 -o runable_casesets_results.csv

```

## Draw paper results:
```bash
python3 plot_20_perf.py
```