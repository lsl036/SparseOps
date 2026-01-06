#!/usr/bin/env python3
"""
Matrix comparison script for SpGEMM results validation.
Compares computed results with reference results.
Since columns are not sorted, we compare row-wise non-zero elements.
"""

import sys
import os
from collections import defaultdict

# Matrix names array for easy extension
MATRIX_NAMES = [
    'bcspwr10',
    'bcsstk32',
    'skirt_id_764'
    # Add more matrix names here in the future
]

# Base directory for matrix files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)


def read_mtx_file(filepath):
    """
    Read a MatrixMarket format file.
    Returns: (rows, cols, nnz, data_dict)
    where data_dict[row] = list of (col, value) tuples
    """
    data_dict = defaultdict(list)
    rows = 0
    cols = 0
    nnz = 0
    
    with open(filepath, 'r') as f:
        # Skip header and comments
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()
        
        # Read dimensions
        parts = line.strip().split()
        if len(parts) >= 3:
            rows = int(parts[0])
            cols = int(parts[1])
            nnz = int(parts[2])
        
        # Read data (MTX uses 1-based indexing)
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                row = int(parts[0]) - 1  # Convert to 0-based
                col = int(parts[1]) - 1  # Convert to 0-based
                value = float(parts[2]) if len(parts) >= 3 else 1.0
                data_dict[row].append((col, value))
    
    return rows, cols, nnz, data_dict


def compare_matrices(ref_file, computed_file, tolerance=1e-10):
    """
    Compare two MTX files.
    Returns: (is_match, error_message)
    """
    print(f"Reading reference file: {ref_file}")
    ref_rows, ref_cols, ref_nnz, ref_data = read_mtx_file(ref_file)
    
    print(f"Reading computed file: {computed_file}")
    comp_rows, comp_cols, comp_nnz, comp_data = read_mtx_file(computed_file)
    
    # Check dimensions
    if ref_rows != comp_rows:
        return False, f"Row dimension mismatch: reference={ref_rows}, computed={comp_rows}"
    if ref_cols != comp_cols:
        return False, f"Column dimension mismatch: reference={ref_cols}, computed={comp_cols}"
    
    # Check total nnz
    if ref_nnz != comp_nnz:
        print(f"Warning: nnz mismatch: reference={ref_nnz}, computed={comp_nnz}")
        print("  (This might be acceptable if there are numerical zeros)")
    
    # Compare row by row
    all_rows = set(ref_data.keys()) | set(comp_data.keys())
    differences = []
    
    for row in sorted(all_rows):
        ref_row_data = ref_data.get(row, [])
        comp_row_data = comp_data.get(row, [])
        
        # Convert to dictionaries for easier comparison
        ref_dict = {col: val for col, val in ref_row_data}
        comp_dict = {col: val for col, val in comp_row_data}
        
        # Check if all columns match
        all_cols = set(ref_dict.keys()) | set(comp_dict.keys())
        
        for col in sorted(all_cols):
            ref_val = ref_dict.get(col, 0.0)
            comp_val = comp_dict.get(col, 0.0)
            
            # Check if column exists in both
            if col not in ref_dict:
                differences.append(f"  Row {row+1}, Col {col+1}: extra in computed (value={comp_val})")
            elif col not in comp_dict:
                differences.append(f"  Row {row+1}, Col {col+1}: missing in computed (value={ref_val})")
            else:
                # Check value difference
                diff = abs(ref_val - comp_val)
                if diff > tolerance:
                    differences.append(
                        f"  Row {row+1}, Col {col+1}: value mismatch "
                        f"(ref={ref_val}, comp={comp_val}, diff={diff})"
                    )
    
    if differences:
        error_msg = f"Found {len(differences)} differences:\n" + "\n".join(differences[:20])
        if len(differences) > 20:
            error_msg += f"\n  ... and {len(differences) - 20} more differences"
        return False, error_msg
    
    return True, "Matrices match!"


def main():
    """Main function to compare matrices."""
    print("=" * 60)
    print("SpGEMM Result Validation")
    print("=" * 60)
    print()
    
    all_passed = True
    
    for matrix_name in MATRIX_NAMES:
        print(f"\n{'='*60}")
        print(f"Testing matrix: {matrix_name}")
        print(f"{'='*60}")
        
        # Construct file paths
        ref_file = os.path.join(SCRIPT_DIR, f"{matrix_name}_res.mtx")
        
        # Try multiple possible locations for computed file
        comp_file = os.path.join(SCRIPT_DIR, f"{matrix_name}_SpOps.mtx")
        if not os.path.exists(comp_file):
            comp_file = os.path.join(BASE_DIR, "build", f"{matrix_name}_SpOps.mtx")
        
        # Check if files exist
        if not os.path.exists(ref_file):
            print(f"ERROR: Reference file not found: {ref_file}")
            all_passed = False
            continue
        
        if not os.path.exists(comp_file):
            print(f"ERROR: Computed file not found: {comp_file}")
            print(f"       Searched in:")
            print(f"         - {os.path.join(SCRIPT_DIR, f'{matrix_name}_SpOps.mtx')}")
            print(f"         - {os.path.join(BASE_DIR, 'build', f'{matrix_name}_SpOps.mtx')}")
            print(f"       Please run test_spgemm first to generate the result.")
            all_passed = False
            continue
        
        # Compare matrices
        is_match, message = compare_matrices(ref_file, comp_file)
        
        if is_match:
            print(f"[PASSED] {matrix_name}")
            print(f"  {message}")
        else:
            print(f"[FAILED] {matrix_name}")
            print(f"  {message}")
            all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

