# FIg11 - Time Overhead

```bash
cd $SPARSEOPS_ROOT/figs/fig11
```
### Count the preprocessing time

Run the testing scripts for 20 matrices:
```bash
python3 $SPARSEOPS_ROOT/script/run_test_spgemm_hc_lsh_list.py $SPARSEOPS_ROOT/runable_casesets.txt  -c CLSH_results_Hyb.csv --threads 128 --kernel 3
```
In paper, we use $genPairs\_time + HC\_time=Time\_Overhead$ . For other reordering methods, we record their reordering time.

### Draw paper Fig11
```
python3 plot_preprocessing_runs.py
```