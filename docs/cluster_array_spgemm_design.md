# Cluster Array-based SpGEMM 实现设计

## 概述

参考 `LeSpGEMM_array_rowwise_new` 的实现，为 cluster 格式设计 array-based SpGEMM 方法。

### 关键差异

1. **Row-wise**: 为每一行收集唯一的列索引
2. **Cluster-wise**: 为每个簇收集唯一的列索引（簇内所有行的列索引的并集）

3. **Row-wise**: 每个列对应一个值
4. **Cluster-wise**: 每个列对应 `cluster_sz` 个值（每个值对应簇内的一行）

## 数据结构

### 输入
- `A_cluster`: `CSR_FlengthCluster` 格式
  - `rowptr[i]`: 簇 i 的列索引在 `colids` 中的起始位置
  - `colids[rowptr[i]..rowptr[i+1]-1]`: 簇 i 的所有列索引
  - `values[(rowptr[i] + j) * cluster_sz + k]`: 簇 i 的第 j 个列，第 k 行的值

- `B`: `CSR_Matrix` 格式（标准 CSR）

### 输出
- `C_cluster`: `CSR_FlengthCluster` 格式（与 A_cluster 结构相同）

## 实现步骤

### 主函数：`LeSpGEMM_array_FLength`

```cpp
template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_array_FLength(
    const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
    const CSR_Matrix<IndexType, ValueType> &B,
    CSR_FlengthCluster<IndexType, ValueType> &C_cluster)
{
    // 1. 创建 BIN 用于负载均衡
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin = 
        new SpGEMM_BIN_FlengthCluster<IndexType, ValueType>(
            A_cluster.rows, A_cluster.cluster_sz, MIN_HT_S);
    
    // 2. 初始化输出矩阵 C_cluster
    C_cluster.csr_rows = A_cluster.csr_rows;
    C_cluster.rows = A_cluster.rows;
    C_cluster.cols = B.num_cols;
    C_cluster.cluster_sz = A_cluster.cluster_sz;
    
    // 3. 设置 BIN（计算每个簇的工作量）
    bin->set_max_bin(A_cluster.rowptr, A_cluster.colids, 
                     B.row_offset, C_cluster.cols);
    
    // 4. 分配簇指针数组
    C_cluster.rowptr = new_array<IndexType>(C_cluster.rows + 1);
    
    // 5. 符号阶段：生成并排序每个簇的列索引
    spgemm_Flength_array_symbolic_new<IndexType, ValueType>(
        A_cluster, B.row_offset, B.col_index,
        C_cluster.rows, C_cluster.cols,
        C_cluster.rowptr, C_cluster.colids, C_cluster.nnzc, bin);
    
    // 6. 分配值数组（每个列需要 cluster_sz 个值）
    C_cluster.values = new_array<ValueType>(
        C_cluster.nnzc * C_cluster.cluster_sz);
    
    // 7. 数值阶段：使用二分查找累加值
    spgemm_Flength_array_numeric_new<sortOutput, IndexType, ValueType>(
        A_cluster, B.row_offset, B.col_index, B.values,
        C_cluster.rows, C_cluster.cols,
        C_cluster.rowptr, C_cluster.colids, C_cluster.values, 
        C_cluster.cluster_sz, bin);
    
    // 8. 设置 Matrix_Features 字段
    C_cluster.num_rows = C_cluster.rows;
    C_cluster.num_cols = C_cluster.cols;
    
    // 9. 清理
    delete bin;
}
```

---

## 函数 1: `spgemm_Flength_array_symbolic_new`

### 功能
为每个簇收集唯一的列索引，排序后存储到 `colids`。

### 实现策略（参考 `spgemm_array_symbolic_new`）
1. **单次遍历**：为每个簇收集唯一的列索引，存储到临时缓冲区
2. **Scan**：计算簇偏移（`rowptr`）
3. **写入**：将存储的列索引复制到 `colids`（已排序）

### 函数签名

```cpp
template <typename IndexType, typename ValueType>
void spgemm_Flength_array_symbolic_new(
    const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_clusters, IndexType c_cols,
    IndexType *crpt, IndexType *&ccolids, IndexType &c_nnzc,
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin)
```

### 关键步骤

1. **分配每簇的临时缓冲区**
   ```cpp
   std::vector<std::vector<IndexType>> cluster_cols(c_clusters);
   ```

2. **并行收集每个簇的唯一列索引**
   ```cpp
   #pragma omp parallel num_threads(bin->allocated_thread_num)
   {
       int tid = Le_get_thread_id();
       IndexType start_cluster = bin->clusters_offset[tid];
       IndexType end_cluster = bin->clusters_offset[tid + 1];
       
       // 找到该线程处理的所有簇中最大的列数
       IndexType max_nz = 0;
       for (IndexType i = start_cluster; i < end_cluster; ++i) {
           if (bin->cluster_nz[i] > max_nz) {
               max_nz = bin->cluster_nz[i];
           }
       }
       if (max_nz > c_cols) max_nz = c_cols;
       
       // 分配临时数组（每个簇重用）
       IndexType *temp_cols = new_array<IndexType>(max_nz);
       
       for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
           IndexType temp_size = 0;
           
           // 获取簇的列范围
           IndexType col_start = A_cluster.rowptr[cluster_id];
           IndexType col_end = A_cluster.rowptr[cluster_id + 1];
           
           // 收集该簇的所有唯一列索引
           // 对于簇中的每个列 j，遍历 B[j, :] 的所有列
           for (IndexType j = col_start; j < col_end; ++j) {
               IndexType t_acol = A_cluster.colids[j];
               // 遍历 B[t_acol, :] 的所有列
               for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                   IndexType key = bcol[k];
                   // 插入并保持排序（使用 insert_if_not_exists）
                   if (insert_if_not_exists(temp_cols, temp_size, max_nz, key)) {
                       // 已插入，temp_cols 保持排序
                   }
               }
           }
           
           // 更新 cluster_nz（该簇的唯一列数）
           bin->cluster_nz[cluster_id] = temp_size;
           
           // 存储该簇的列索引（已排序）
           cluster_cols[cluster_id].resize(temp_size);
           for (IndexType j = 0; j < temp_size; ++j) {
               cluster_cols[cluster_id][j] = temp_cols[j];
           }
       }
       
       delete_array(temp_cols);
   }
   ```

3. **Scan：计算簇偏移**
   ```cpp
   scan(bin->cluster_nz, crpt, c_clusters + 1, bin->allocated_thread_num);
   c_nnzc = crpt[c_clusters];
   ```

4. **分配 colids 数组**
   ```cpp
   ccolids = new_array<IndexType>(c_nnzc);
   ```

5. **写入存储的列索引到 colids**
   ```cpp
   #pragma omp parallel num_threads(bin->allocated_thread_num)
   {
       int tid = Le_get_thread_id();
       IndexType start_cluster = bin->clusters_offset[tid];
       IndexType end_cluster = bin->clusters_offset[tid + 1];
       
       for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
           IndexType cluster_start = crpt[cluster_id];
           IndexType cluster_nnz = bin->cluster_nz[cluster_id];
           
           // 复制已排序的列索引到输出
           for (IndexType j = 0; j < cluster_nnz; ++j) {
               ccolids[cluster_start + j] = cluster_cols[cluster_id][j];
           }
       }
   }
   ```

---

## 函数 2: `spgemm_Flength_array_numeric_new`

### 功能
使用预排序的 `colids`，对簇内的每一行使用二分查找累加值。

### 实现策略（参考 `spgemm_array_numeric_new`）
1. **初始化**：将 `cvalues` 初始化为 0
2. **遍历簇**：对每个簇，遍历簇内的每一行
3. **二分查找**：对每个中间乘积，使用二分查找在预排序的 `colids` 中找到位置
4. **累加**：将值累加到对应的位置（考虑簇内行索引）

### 函数签名

```cpp
template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_Flength_array_numeric_new(
    const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_clusters, IndexType c_cols,
    const IndexType *crpt, const IndexType *ccolids, ValueType *cvalues,
    IndexType cluster_sz,
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin)
```

### 关键步骤

1. **初始化值数组为 0**
   ```cpp
   #pragma omp parallel num_threads(bin->allocated_thread_num)
   {
       int tid = Le_get_thread_id();
       IndexType start_cluster = bin->clusters_offset[tid];
       IndexType end_cluster = bin->clusters_offset[tid + 1];
       
       for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
           IndexType cluster_start = crpt[cluster_id];
           IndexType cluster_nnz = crpt[cluster_id + 1] - cluster_start;
           if (cluster_nnz > 0) {
               // 初始化该簇的所有值（cluster_nnz * cluster_sz 个值）
               std::memset(cvalues + cluster_start * cluster_sz, 0, 
                          cluster_nnz * cluster_sz * sizeof(ValueType));
           }
       }
   }
   ```

2. **累加中间乘积**
   ```cpp
   #pragma omp parallel num_threads(bin->allocated_thread_num)
   {
       int tid = Le_get_thread_id();
       IndexType start_cluster = bin->clusters_offset[tid];
       IndexType end_cluster = bin->clusters_offset[tid + 1];
       
       for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
           IndexType cluster_start = crpt[cluster_id];
           IndexType cluster_nnz = crpt[cluster_id + 1] - cluster_start;
           if (cluster_nnz == 0) continue;
           
           // 获取该簇的列索引（已排序）
           const IndexType *cluster_ccolids = ccolids + cluster_start;
           
           // 获取该簇在 A 中的列范围
           IndexType a_col_start = A_cluster.rowptr[cluster_id];
           IndexType a_col_end = A_cluster.rowptr[cluster_id + 1];
           
           // 对簇内的每一行进行处理
           for (IndexType row_in_cluster = 0; row_in_cluster < cluster_sz; ++row_in_cluster) {
               // 计算该行在原始 CSR 中的行号
               IndexType csr_row = cluster_id * cluster_sz + row_in_cluster;
               
               // 如果该行超出了原始矩阵的行数，跳过
               if (csr_row >= A_cluster.csr_rows) break;
               
               // 遍历该簇的所有列（这些列对应簇内所有行的非零元素）
               for (IndexType j = a_col_start; j < a_col_end; ++j) {
                   IndexType t_acol = A_cluster.colids[j];
                   
                   // 获取该行在该列的值
                   ValueType t_aval = A_cluster.values[j * cluster_sz + row_in_cluster];
                   
                   // 如果值为 0，跳过（稀疏矩阵优化）
                   if (std::abs(t_aval) < eps) continue;
                   
                   // 遍历 B[t_acol, :] 的所有列
                   for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                       IndexType target_col = bcol[k];
                       ValueType product = t_aval * bval[k];
                       
                       // 二分查找在预排序的 cluster_ccolids 中找到位置
                       IndexType pos = binary_search_find(
                           cluster_ccolids, cluster_nnz, target_col);
                       
                       // 如果找到，累加到对应的位置
                       // 位置计算：cluster_start * cluster_sz + pos * cluster_sz + row_in_cluster
                       if (pos != -1) {
                           IndexType val_idx = (cluster_start + pos) * cluster_sz + row_in_cluster;
                           cvalues[val_idx] += product;
                       }
                   }
               }
           }
       }
   }
   ```

---

## 辅助函数

### `binary_search_find`（已存在）
在排序数组中查找元素，返回位置或 -1。

### `insert_if_not_exists`（已存在）
在排序数组中插入元素（如果不存在），保持数组排序。

---

## 关键优化点

1. **单次遍历符号阶段**：避免重复遍历 A 和 B
2. **预排序列索引**：数值阶段只需二分查找，无需插入
3. **簇级负载均衡**：使用 `SpGEMM_BIN_FlengthCluster` 进行负载均衡
4. **内存局部性**：簇格式提高缓存利用率

---

## 与 Hash 方法的对比

| 特性 | Hash 方法 | Array 方法 |
|------|-----------|------------|
| 符号阶段 | Hash 表收集列索引 | 排序数组收集列索引 |
| 数值阶段 | Hash 表查找和累加 | 二分查找和累加 |
| 内存使用 | 需要 Hash 表（2^N 大小） | 精确大小（无填充） |
| 排序 | 需要额外排序步骤 | 自然排序（符号阶段已排序） |
| 查找复杂度 | O(1) 平均 | O(log n) |
| 插入复杂度 | O(1) 平均 | N/A（预分配） |

---

## 待实现文件

1. **头文件**: `include/spgemm_Flength_array.h`
   - 声明 `spgemm_Flength_array_symbolic_new`
   - 声明 `spgemm_Flength_array_numeric_new`

2. **实现文件**: `src/spgemm_Flength_array.cpp`
   - 实现上述两个函数

3. **主接口**: `src/spgemm.cpp`
   - 实现 `LeSpGEMM_array_FLength`
   - 在 `LeSpGEMM_FLength` 中添加 `kernel_flag=2` 的分支
