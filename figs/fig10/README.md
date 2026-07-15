# Fig10- Ablation Study of Reordering
## 0. Preparation
```bash
cd $SPARSEOPS_ROOT/figs/fig10
```
To test different reordering methods (such as hp, gp, gray order), the reordering results need to be pre- downloaded by :
```bash
python3 $SPARSEOPS_ROOT/data/download_datasets.py --reordering
```

## 1. Quick Test for 4 Reordering Methods

### 1.1 Quick test for each methods
```bash
# gray order
$SPARSEOPS_ROOT/build/test_reordered_spgemm_hc_lsh $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/reordering/gray_order/cant.grayorder --kernel=3 --threads=128

# RCM order
$SPARSEOPS_ROOT/build/test_reordered_spgemm_hc_lsh $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/reordering/rcm_order/cant.rcmorder --kernel=3 --threads=128

# GP order
$SPARSEOPS_ROOT/build/test_reordered_spgemm_hc_lsh $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/reordering/gp_order/cant.gporder --kernel=3 --threads=128

# HP order
$SPARSEOPS_ROOT/build/test_reordered_spgemm_hc_lsh $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/reordering/hp_order/cant.hporder --kernel=3 --threads=128
```
### 1.2 Test scripts
1. For hybrid accumulator performance:
```bash
KERNEL=3 bash $SPARSEOPS_ROOT/script/run_test_reordered_spgemm_hc_lsh.sh rcm hp gp gray
```
2. For hash-based accumulator performance:
```bash
KERNEL=1 bash $SPARSEOPS_ROOT/script/run_test_reordered_spgemm_hc_lsh.sh rcm hp gp gray
```

## 2. Hierarchical AAT Reordering
### 2.1 Quick test for AAT reordering SpGEMM
``` bash
$SPARSEOPS_ROOT/build/test_spgemm_hc $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/reordering/close_pair/cant.mtx --kernel=3 --threads=128
```
### 2.2 Test scripts
1. For hybrid accumulator performance:
```bash
BASE_DIR=$SPARSEOPS_ROOT/data CLOSE_PAIRS_DIR=$SPARSEOPS_ROOT/data/reordering/close_pair THREADS=128 KERNEL=3 RESULT_CSV=$SPARSEOPS_ROOT/figs/fig10/AAtHYB_results.csv bash $SPARSEOPS_ROOT/script/run_test_spgemm_hc.sh
```
2. For hash-based accumulator performance:
```bash
BASE_DIR=$SPARSEOPS_ROOT/data CLOSE_PAIRS_DIR=$SPARSEOPS_ROOT/data/reordering/close_pair THREADS=128 KERNEL=1 RESULT_CSV=$SPARSEOPS_ROOT/figs/fig10/AAtHash_results.csv bash $SPARSEOPS_ROOT/script/run_test_spgemm_hc.sh
```

## 3. Naive LSH Reordering

### 3.1 Quick test 
```bash
$SPARSEOPS_ROOT/build/test_spgemm_hc_lsh $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/cant/cant.mtx --hc_v=2 --kernel=3 --threads=128
```
### 3.2 Test scripts
1. For hybrid accumulator performance:
```bash
python3 $SPARSEOPS_ROOT/script/run_test_spgemm_hc_lsh_list.py $SPARSEOPS_ROOT/runable_casesets.txt --hc_v=2 -c naiveLSH_results_Hyb.csv --threads 128 --kernel 3
```
2. For hash-based accumulator performance:
```bash
python3 $SPARSEOPS_ROOT/script/run_test_spgemm_hc_lsh_list.py $SPARSEOPS_ROOT/runable_casesets.txt --hc_v=2 -c naiveLSH_results_Hash.csv --threads 128 --kernel 1
```
## 4. C-LSH Reordering (LeSpGEMM)
### 4.1 Quick test 
```bash
$SPARSEOPS_ROOT/build/test_spgemm_hc_lsh $SPARSEOPS_ROOT/data/cant/cant.mtx $SPARSEOPS_ROOT/data/cant/cant.mtx --hc_v=0 --kernel=3 --threads=128
```
### 4.2 Test scripts
1. For hybrid accumulator performance:
```bash
python3 $SPARSEOPS_ROOT/script/run_test_spgemm_hc_lsh_list.py $SPARSEOPS_ROOT/runable_casesets.txt  -c CLSH_results_Hyb.csv --threads 128 --kernel 3
```
2. For hash-based accumulator performance:
```bash
python3 $SPARSEOPS_ROOT/script/run_test_spgemm_hc_lsh_list.py $SPARSEOPS_ROOT/runable_casesets.txt  -c CLSH_results_Hash.csv --threads 128 --kernel 1
```

## 5. Draw paper figure

Using our results to reproduce Figure 10 in paper :
```
python3 plot_reordering_ablation.py
```