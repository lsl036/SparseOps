# Fig12 - Memory Overhead Analysis
```bash
cd $SPARSEOPS_ROOT/figs/fig12
```

## Record Max Memory Usage
We record the max memory usage when doing preprocessing of C-LSH clustering step of LeSpGEMM, run script:

- For our C-LSH algorithm:
```bash
python3 $SPARSEOPS_ROOT/script/run_record_permutation_lsh_list.py $SPARSEOPS_ROOT/runable_casesets.txt --base-dir $SPARSEOPS_ROOT/data --out-dir $SPARSEOPS_ROOT/data/reordering/lsh_order -c CLSH_Mem_Overhead.csv --record-maxmem
```
- For naive LSH algorithm:
```bash
python3 $SPARSEOPS_ROOT/script/run_record_permutation_lsh_list.py $SPARSEOPS_ROOT/runable_casesets.txt --hc-v 2 --base-dir $SPARSEOPS_ROOT/data --out-dir $SPARSEOPS_ROOT/data/reordering/naivelsh_order -c NaiveLSH_Mem_Overhead.csv --record-maxmem
```

## Draw paper Fig12
```bash
python3 plot_preprocessing_memory.py
```