# Fig13 - Breakdown

```bash
cd $SPARSEOPS_ROOT/figs/fig12
```

## Record the time breakdown of LeSpGEMM

```bash
python3 $SPARSEOPS_ROOT/script/run_test_spgemm_hc_lsh_list.py $SPARSEOPS_ROOT/runable_casesets.txt --base-dir $SPARSEOPS_ROOT/data --print-bd --kernel 3 -c breakdown.csv
```

## Draw paper Fig13
```
python3 plot_runtime_breakdown.py
```