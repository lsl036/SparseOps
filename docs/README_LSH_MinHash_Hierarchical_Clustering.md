# MinHash + LSH 与层次聚类在 SpGEMM 行聚类中的策略

本文档按当前 SparseOps 中 **v1** 的代码逻辑，总结用于稀疏矩阵乘法（SpGEMM）行聚类的 **MinHash → LSH 候选对 → 层次聚类** 全流程，便于复现与论文写作参考。

---

## 1. 整体流程

```
输入: 矩阵 A (CSR), 参数 k, num_bands, cluster_size, seed
  ↓
Phase 1: MinHash 签名 → 每行一个 k 维签名 (flat 存储)
  ↓
Phase 2: LSH Banding → 按 band 哈希进桶，同桶行成候选对，去重后得到 (i,j) 及 MinHash 估计 Jaccard
  ↓
Phase 3: 层次聚类 (v1) → 优先队列按相似度合并，合并时发现新根对则用精确 Jaccard 入队，无 map
  ↓
输出: permutation, offsets → 行重排 + 变长簇划分，供 VLength-Cluster SpGEMM 使用
```

目标：在**低预处理开销**下，将**行结构相似**（Jaccard 高）的行聚成簇，以便后续用变长簇 SpGEMM 核提高缓存命中与计算密度。

---

## 2. Phase 1：MinHash 签名

### 2.1 数学与语义

- 将矩阵 **A** 的每一行视为**列索引集合** $S_i$（若采用 CSR，可忽略重复列或视为多重集取唯一）。
- **Jaccard 相似度**：
  $
  J(i,j) = \frac{|S_i \cap S_j|}{|S_i \cup S_j|}.
  $
- **MinHash**：使用 $k$ 个独立哈希 $(h_1,\ldots,h_k)$，对行 \(i\) 的签名
  $[
  \sigma_i[d] = \min_{c \in S_i} h_d(c), \quad d=1,\ldots,k.
  ]$
- 在常用哈希假设下，$\mathbb{P}(\sigma_i[d] = \sigma_j[d]) = J(i,j)$，故
  $
  \widehat{J}(i,j) = \frac{1}{k}\sum_{d=1}^{k} \mathbb{1}[\sigma_i[d] = \sigma_j[d]]
  $
  是 Jaccard 的无偏估计。

### 2.2 实现要点（与 v1 一致）

- **哈希族**：线性同余 $h_d(x) = a_d \cdot x + b_d \pmod{2^{64}}$，\(a_d\) 为奇数，由 LCG 从 `seed` 生成 \((a_d, b_d)\)。
- **存储**：`minhash_signatures_flat`，单一大数组 `sigs[0..num_rows*k-1]`，行 \(i\) 的签名在 `sigs[i*k .. (i+1)*k - 1]`，便于后续 LSH 和向量化。
- **并行**：按行并行计算 MinHash；内层对 \(k\) 维可做 4/8 路循环展开或 AVX-512（8 个 64-bit 最小值的更新与比较）。
- **空行**：签名全为 `UINT64_MAX`，估计 Jaccard 时两空行在对应维上视为相等。

---

## 3. Phase 2：LSH Banding 与候选对

### 3.1 Banding 规则

- 将 \(k\) 维分成 `num_bands` 段，每段长度 \(r = k / \texttt{num_bands}\)（要求 \(k \equiv 0 \pmod{\texttt{num_bands}}\)）。
- 对每个 band \(b = 0,\ldots,\texttt{num_bands}-1\)，行 \(i\) 在该 band 的 \(r\) 个 MinHash 值通过 `band_hash` 映射为一个桶 ID。
- **候选对定义**：若存在至少一个 band，两行落入同一桶，则 \((i,j)\) 为候选对；同一候选对可能来自多个 band，去重后只保留一份。

### 3.2 Band 哈希与桶

- `band_hash(band_sig, r)`：对长度为 \(r\) 的 MinHash 子向量做一次 64-bit 混合哈希（如乘加组合），得到桶 ID。
- 每个 band 独立建一个 `unordered_map<uint64_t, vector<IndexType>>`：桶 ID → 行下标列表；并行时每个线程负责若干 band，最后合并候选对。

### 3.3 候选对生成与去重

- 小桶（大小 \(\le\) `k_bucket_size_limit`，如 256）：桶内所有行两两成对 \((i,j)\)，并规范为 \(i < j\)。
- 大桶：为控制 \(O(n^2)\)，仅对每个行 \(a\) 与其后**最多** `k_window_pairs`（如 16）个行 \(b\) 成对，避免候选数爆炸。
- 所有 band 的候选对合并后，按 \((i,j)\) **排序并 unique**，得到无重复候选对列表。

### 3.4 相似度分数

- 对每个去重后的候选对 \((i,j)\)，用 **MinHash 估计 Jaccard** 作为初始分数：
  \[
  \texttt{score}(i,j) = \frac{1}{k}\sum_{d=1}^{k} \mathbb{1}[\sigma_i[d] = \sigma_j[d]].
  \]
- 实现上可用标量循环或 AVX-512 比较 8 个 64-bit 对并统计相等数，再除以 \(k\)。
- **v1 接口**：输出为 `vector<CandidatePair>`，即 `(i, j, score)` 的线性数组，**不**使用 `std::map`，以减少预处理开销与缓存不友好访问。

---

## 4. Phase 3：层次聚类（v1 逻辑）

### 4.1 目标

- 输入：候选对集合（含 MinHash 估计分数）、`cluster_size`（簇最大行数）、矩阵 A 的 CSR 以支持按需精确 Jaccard。
- 输出：`permutation`（新行号 → 原行号）、`offsets`（每个簇的起始下标），使同一簇内行在重排后连续，且簇大小不超过 `cluster_size`。

### 4.2 数据结构（无 map）

- **并查集**：`clusters[]`、`sz[]`，带 **路径压缩** 的 `find(x)`，返回根并压缩路径上所有节点指向根。
- **优先队列**：`(score, (i,j))` 按 score **降序**（大根堆）；比较器在 score 相同时按 \((i,j)\) 字典序 tie-break，保证与 v0 一致的合并顺序。
- **已见集合**：`set<pair<IndexType,IndexType>>` 记录已入队的根对 \((r_i,r_j)\)（规范为 first < second），用于“合并时发现新根对”的去重，**不存 score**，避免使用 map。

### 4.3 主循环（与 v0 算法对齐）

1. **初始化**：所有候选对 \((i,j)\) 规范为 \(i<j\)，若未在 `seen` 中则加入 `seen` 并压入优先队列。
2. **循环**：当队列非空且仍有可合并簇时：
   - 弹出当前最高分的 \((i,j)\)；
   - \(r_i = \texttt{find}(i),\; r_j = \texttt{find}(j)\)；
   - 若 \(r_i = r_j\) 或任一根已无效（`valid[root]=0`），则跳过；
   - **若当前 \(i,j\) 仍是根**（`clusters[i]==i` 且 `clusters[j]==j`）：按秩合并两簇；若合并后根的大小 \(\ge\) `cluster_size`，将该根标记为无效并减少“活跃簇”计数；
   - **否则**（至少一方已被合并过）：得到的是“新根对” \((r_i,r_j)\)；若 \((r_i,r_j)\) 规范化为 \((p_i,p_j)\) 后不在 `seen` 中，则计算**精确 Jaccard**（基于 CSR 的列集合交并），将 \((p_i,p_j)\) 入队并加入 `seen`。
3. **输出**：根据并查集得到每个元素的根，统计各根对应簇大小，按根编号顺序写出 `offsets`；在同一簇内按原行号顺序写出 `permutation`（新下标 → 原行号）。

### 4.4 与 v0 的对应关系

- v0 使用 `map` 存 (根对 → score)，并在“合并时发现根对”时插入新项；v1 用 **set 只记“已见根对”** + 优先队列中的 (score, 根对)，**不存 score 的 map**，预处理与内存更轻。
- 比较器、合并规则、按秩合并与 `cluster_size`/`valid` 逻辑与 v0 一致，从而在相同输入下得到**相同的行聚类结果**（同一批行属于同一簇）。

---

## 5. 与 SpGEMM 的衔接

- 用 `permutation` 对 A 的行做重排得到 `A_reordered`，再根据 `offsets` 转为 **CSR_VlengthCluster**（变长簇格式）。
- 后续执行 **LeSpGEMM_VLength** 等变长簇 SpGEMM 核，实现“基于 MinHash+LSH+层次聚类的行聚类 SpGEMM”流程。

---

## 6. 参数小结

| 参数 | 含义 | 典型取值 |
|------|------|----------|
| `k` | MinHash 签名长度 | 64 |
| `num_bands` | LSH band 数，要求 \(k \bmod \texttt{num\_bands}=0\) | 16 |
| `r = k/num_bands` | 每 band 维数 | 4 |
| `seed` | MinHash 哈希族随机种子 | 7 或 12345 |
| `cluster_size` | 层次聚类单簇最大行数 | 8 |
| `k_bucket_size_limit` | 桶内全配对的大小上限 | 256 |
| `k_window_pairs` | 大桶内每行最多配对数 | 16 |

---

## 7. 创新点总结（供论文写作参考）

以下要点可直接用于方法章节与贡献总结，按“问题 → 做法 → 效果”组织。

### 7.1 面向 SpGEMM 行聚类的 MinHash–LSH–HC 流水线

- **问题**：精确计算所有行对的 Jaccard 为 \(O(n^2 \cdot \bar{m})\)，不可扩展；需要快速得到“高相似行对”用于聚类。
- **做法**：用 MinHash 将行压缩为定长签名，用 LSH banding 在近似 \(O(n \cdot k)\) 时间内得到候选对，再用 MinHash 估计 Jaccard 作为层次聚类的初始分数；仅在层次聚类合并过程中对“新出现的根对”按需计算精确 Jaccard。
- **效果**：预处理从全对精确相似度降为 MinHash+LSH+有限次精确 Jaccard，在保持与全对精确聚类一致语义的前提下，显著降低预处理成本与内存。

### 7.2 无 map 的层次聚类实现（v1）与“合并时发现根对”

- **问题**：经典实现用 map 维护 (根对 → 相似度)，插入与查找开销大，且不利于缓存。
- **做法**：用 **set 仅记录已入队的根对**，用**优先队列**维护 (相似度, 根对)；合并时若发现新根对且未在 set 中，则计算其精确 Jaccard 并压入队列，同时加入 set，**不再使用 map**。
- **效果**：算法行为与 v0（map 版）一致（相同比较器与合并规则），在更短预处理开销与更少随机写的前提下，达到相同行聚类结果，便于工程部署与对比实验。

### 7.3 LSH 大桶的固定窗口采样

- **问题**：LSH 某 band 中桶过大时，桶内两两成对为 \(O(n^2)\)，会拖慢候选对生成。
- **做法**：设定桶大小上限（如 256）；超过时对桶内行只做**固定窗口采样**：每个行只与其后最多 \(w\) 个行（如 16）成对，总候选数 \(O(n \cdot w)\)。
- **效果**：在保留高相似行对进入候选的前提下，控制最坏情况复杂度与运行时间，避免个别大桶主导整体耗时。

### 7.4 扁平 MinHash 存储与向量化估计 Jaccard

- **问题**：二维 `vector<vector<uint64_t>>` 签名不利于连续访问与 SIMD。
- **做法**：采用 **row-major 扁平数组** `sigs[row*k .. (row+1)*k-1]` 存储 MinHash；在 MinHash 计算与候选对的 Jaccard 估计中，对 \(k\) 维使用 4/8 路展开或 AVX-512（8×64-bit 比较与计数）。
- **效果**：更好的缓存局部性与向量化利用率，加速 Phase 1 与 Phase 2 的分数计算，适合作为 SpGEMM 预处理的轻量级前端。

### 7.5 确定性 tie-break 与 v0/v1 结果对齐

- **问题**：优先队列中多对具有相同估计/精确 Jaccard 时，弹出顺序未定义会导致不同运行或不同实现得到不同簇划分。
- **做法**：在比较器中增加 **第二键**：当 score 相等时，按 \((i,j)\) 字典序比较，使合并顺序确定。
- **效果**：v0 与 v1 在相同输入下得到**完全一致的行聚类结果**，便于复现与公平对比（如与 map 版、与其它聚类方法）。

### 7.6 精确 Jaccard 的按需计算与混合策略

- **问题**：全部候选对都算精确 Jaccard 代价高；仅用 MinHash 估计又可能在某些对上偏差较大。
- **做法**：层次聚类中**仅对“合并时新发现的根对”**计算精确 Jaccard 并入队，其余仍用 MinHash 估计或初始分数；根对去重用 set，避免重复计算。
- **效果**：在保证聚类质量（新根对用精确相似度决策）的前提下，将精确 Jaccard 调用次数限制在“合并过程中实际出现的新根对”数量级，通常远小于候选对总数。

---

以上内容与当前 `spgemm_MinHashLSH.h` 及 `spgemm_utility.h` 中 v1 实现一致，可直接作为技术文档与论文方法部分的参考。若后续实现有参数或接口变更，只需同步更新本 README 与创新点列表即可。
