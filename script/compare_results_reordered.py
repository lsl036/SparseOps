#!/usr/bin/env python3
"""
Matrix comparison script for Reordered SpGEMM results validation.
Compares computed results with reference results.
Supports kernel 1 (hash-based), kernel 2 (optimized array-based), and kernel 3 (SPA-based).

Usage:
    python3 compare_results_reordered.py [--kernel=1|2|3|hashrowwise|arrayrowwise|sparowwise]
    
    --kernel: Select kernel to compare (default: 1)
              1 or hashrowwise: Hash-based row-wise kernel
              2 or arrayrowwise: Optimized array-based row-wise kernel
              3 or sparowwise: SPA-based array row-wise kernel
"""

import sys
import os
import argparse
from collections import defaultdict

# Matrix name for reordered SpGEMM test
MATRIX_NAME = '2cubes_sphere'

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


def compare_matrices(ref_file, computed_file, rel_tolerance=1e-12, abs_tolerance=1e-10):
    """
    Compare two MTX files using relative error for large values and absolute error for small values.
    
    Args:
        ref_file: Path to reference MTX file
        computed_file: Path to computed MTX file
        rel_tolerance: Relative error tolerance for large values (default: 1e-12)
        abs_tolerance: Absolute error tolerance for small values (default: 1e-10)
    
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
                # Check value difference using relative error for large values
                abs_diff = abs(ref_val - comp_val)
                
                # Use relative error for large values, absolute error for small values
                if abs(ref_val) > 1.0:
                    # For large values, use relative error
                    rel_diff = abs_diff / abs(ref_val)
                    if rel_diff > rel_tolerance:
                        differences.append(
                            f"  Row {row+1}, Col {col+1}: value mismatch "
                            f"(ref={ref_val}, comp={comp_val}, abs_diff={abs_diff:.2e}, rel_diff={rel_diff:.2e})"
                        )
                else:
                    # For small values (including zero), use absolute error
                    if abs_diff > abs_tolerance:
                        differences.append(
                            f"  Row {row+1}, Col {col+1}: value mismatch "
                            f"(ref={ref_val}, comp={comp_val}, abs_diff={abs_diff:.2e})"
                        )
    
    if differences:
        error_msg = f"Found {len(differences)} differences:\n" + "\n".join(differences[:20])
        if len(differences) > 20:
            error_msg += f"\n  ... and {len(differences) - 20} more differences"
        return False, error_msg
    
    return True, "Matrices match!"


def parse_kernel_arg(kernel_arg):
    """
    Parse kernel argument and return suffix string.
    Returns: suffix string (e.g., "hashrowwise", "arrayrowwise", or "sparowwise")
    """
    if kernel_arg is None:
        return "hashrowwise"  # Default to kernel 1
    
    kernel_arg = str(kernel_arg).lower()
    
    # Support numeric values
    if kernel_arg == "1":
        return "hashrowwise"
    elif kernel_arg == "2":
        return "arrayrowwise"
    elif kernel_arg == "3":
        return "sparowwise"
    # Support string values
    elif kernel_arg == "hashrowwise" or kernel_arg == "hash":
        return "hashrowwise"
    elif kernel_arg == "arrayrowwise" or kernel_arg == "array":
        return "arrayrowwise"
    elif kernel_arg == "sparowwise" or kernel_arg == "spa":
        return "sparowwise"
    else:
        print(f"Warning: Unknown kernel argument '{kernel_arg}', defaulting to hashrowwise")
        return "hashrowwise"


def get_kernel_display_name(suffix):
    """Get human-readable kernel name from suffix."""
    if suffix == "hashrowwise":
        return "Hash-based row-wise"
    elif suffix == "arrayrowwise":
        return "Optimized array-based row-wise"
    elif suffix == "sparowwise":
        return "SPA-based array row-wise"
    else:
        return f"Unknown ({suffix})"


def main():
    """Main function to compare matrices."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compare Reordered SpGEMM computed results with reference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 compare_results_reordered.py --kernel=1
  python3 compare_results_reordered.py --kernel=hashrowwise
  python3 compare_results_reordered.py --kernel=2
  python3 compare_results_reordered.py --kernel=arrayrowwise
  python3 compare_results_reordered.py --kernel=3
  python3 compare_results_reordered.py --kernel=sparowwise
        """
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="1",
        help="Kernel to compare: 1 or hashrowwise (Hash-based row-wise, default), "
             "2 or arrayrowwise (Optimized array-based row-wise), "
             "3 or sparowwise (SPA-based array row-wise)"
    )
    
    args = parser.parse_args()
    kernel_suffix = parse_kernel_arg(args.kernel)
    kernel_display = get_kernel_display_name(kernel_suffix)
    
    print("=" * 60)
    print("Reordered SpGEMM Result Validation")
    print("=" * 60)
    print(f"Matrix: {MATRIX_NAME}")
    print(f"Kernel: {kernel_display} (suffix: {kernel_suffix})")
    print()
    
    # Construct file paths
    # Reference file: [name]_ROres.mtx
    ref_file = os.path.join(SCRIPT_DIR, f"{MATRIX_NAME}_ROres.mtx")
    
    # Computed file: [name]_SpOps_reordered_[kernel_suffix].mtx
    comp_filename = f"{MATRIX_NAME}_SpOps_reordered_{kernel_suffix}.mtx"
    
    # Try multiple possible locations for computed file
    comp_file = os.path.join(SCRIPT_DIR, comp_filename)
    if not os.path.exists(comp_file):
        comp_file = os.path.join(BASE_DIR, "build", comp_filename)
    
    # Check if files exist
    if not os.path.exists(ref_file):
        print(f"ERROR: Reference file not found: {ref_file}")
        print(f"       Please ensure the reference file exists.")
        return 1
    
    if not os.path.exists(comp_file):
        print(f"ERROR: Computed file not found: {comp_filename}")
        print(f"       Searched in:")
        print(f"         - {os.path.join(SCRIPT_DIR, comp_filename)}")
        print(f"         - {os.path.join(BASE_DIR, 'build', comp_filename)}")
        print(f"       Please run test_reordered_spgemm with --kernel={args.kernel} first to generate the result.")
        return 1
    
    # Compare matrices
    print(f"\n{'='*60}")
    print(f"Testing matrix: {MATRIX_NAME}")
    print(f"Kernel: {kernel_display}")
    print(f"{'='*60}")
    
    is_match, message = compare_matrices(ref_file, comp_file)
    
    print()
    print("=" * 60)
    if is_match:
        print(f"[PASSED] {MATRIX_NAME} ({kernel_display})")
        print(f"  {message}")
        return 0
    else:
        print(f"[FAILED] {MATRIX_NAME} ({kernel_display})")
        print(f"  {message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
