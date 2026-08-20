
# Fig8 - 600+ Datasets Results

## Download All Matrices

```
cd $SPARSEOPS_ROOT
python3 data/download_datasets.py --source all
```

## Run 600+ matrices performance test

```bash
cd $SPARSEOPS_ROOT/figs/fig8/

python3 $SPARSEOPS_ROOT/script/run_test_spgemm_hc_lsh_list.py $SPARSEOPS_ROOT/runable_datasets.txt --base-dir $SPARSEOPS_ROOT/data --kernel 3 -o runable_datasets_results.csv

```

the results will be recored at `$SPARSEOPS_ROOT/runable_datasets_results.csv`

### Draw paper results:
```bash
python3 plot_performance_3platform.py
```