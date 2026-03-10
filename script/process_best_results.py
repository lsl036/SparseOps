#!/usr/bin/env python3
"""
处理 实验结果.xlsx：对每个 sheet，按 mtxname 保留 LeSpGEMM_VLength_time 最小的那一行，
写入新表格（同样 3 个 sheet，每个数据集只保留性能最好的那一行）。
"""

import pandas as pd

INPUT_FILE = "600实验结果.xlsx"
OUTPUT_FILE = "600实验结果_最佳行.xlsx"

#对自己的 SparseOps 整理最好的实验结果
def main():
    xl = pd.ExcelFile(INPUT_FILE)
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
            if "mtxname" not in df.columns or "LeSpGEMM_VLength_time" not in df.columns:
                print(f"  [跳过] {sheet_name}: 缺少 mtxname 或 LeSpGEMM_VLength_time 列")
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                continue
            # 只对有效 mtxname 做分组；空行（mtxname 为空）不参与“取最优”，最后原样保留
            df_valid = df.dropna(subset=["mtxname"])
            empty_mtx = df[df["mtxname"].isna()]
            if df_valid.empty:
                best_df = df
            else:
                idx_best = df_valid.groupby("mtxname", sort=False)["LeSpGEMM_VLength_time"].idxmin()
                idx_best = idx_best.dropna()  # 避免全空列导致 idxmin 为 NaN 报 KeyError
                if idx_best.empty:
                    best_df = df_valid
                else:
                    best_df = df.loc[idx_best].reset_index(drop=True)
                if not empty_mtx.empty:
                    best_df = pd.concat([best_df, empty_mtx], ignore_index=True)
            best_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  {sheet_name}: {len(df)} 行 -> {len(best_df)} 行（每 mtxname 保留最优）")
    print(f"已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
