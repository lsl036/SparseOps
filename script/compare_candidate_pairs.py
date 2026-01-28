#!/usr/bin/env python3
"""
Candidate pairs comparison script for HashSpGEMMTopK results validation.
Compares computed candidate pairs with reference results.

Usage:
    python3 compare_candidate_pairs.py [--topk=7] [--ref_dir=...] [--computed_dir=...]
    
    --topk: Top-k value used (default: 7)
    --ref_dir: Directory containing reference results (default: /data2/linshengle_data/SpGEMM-Reordering/close_pairs/)
    --computed_dir: Directory containing computed results (default: script/)
"""

import sys
import os
import argparse
from collections import defaultdict

# Matrix names to compare
MATRIX_NAMES = [
    'poisson3Da',
    'cant',
    'pdb1HYS'
]

# Default directories
DEFAULT_REF_DIR = '/data2/linshengle_data/SpGEMM-Reordering/close_pairs'
DEFAULT_COMPUTED_DIR = 'script'
DEFAULT_TOPK = 7


def read_mtx_file(filepath):
    """
    Read a MatrixMarket format file or simple coordinate format.
    Supports two formats:
    1. Standard MTX format with header (%%MatrixMarket ...)
    2. Simple coordinate format without header (row col value)
    
    Returns: (rows, cols, nnz, data_dict)
    where data_dict[row] = sorted list of (col, value) tuples
    """
    data_dict = defaultdict(list)
    rows = 0
    cols = 0
    nnz = 0
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        
        # Check if it's standard MTX format
        if first_line.startswith('%%MatrixMarket'):
            # Standard MTX format - skip header and comments
            line = first_line
            while line.startswith('%'):
                line = f.readline().strip()
            
            # Read dimensions
            parts = line.split()
            if len(parts) >= 3:
                rows = int(parts[0])
                cols = int(parts[1])
                nnz = int(parts[2])
            
            # Read data (detect 0-based or 1-based indexing)
            # Reference code uses 0-based indexing, standard MTX uses 1-based
            # Detect by checking if first data line (after dimensions) contains 0
            first_data_line = True
            is_zero_based = None
            dimension_line_read = False
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        row_raw = int(parts[0])
                        col_raw = int(parts[1])
                        
                        # Skip dimension line (three integers, no decimal point)
                        if not dimension_line_read and len(parts) >= 3:
                            try:
                                nnz_val = int(parts[2])
                                # This looks like a dimension line
                                dimension_line_read = True
                                continue
                            except ValueError:
                                pass
                        
                        dimension_line_read = True
                        
                        # Detect indexing format on first actual data line
                        if first_data_line and is_zero_based is None:
                            # If we see 0 in first data line, it's likely 0-based
                            # Standard MTX would start with 1
                            is_zero_based = (row_raw == 0 or col_raw == 0)
                            first_data_line = False
                        
                        # Convert based on detected format
                        if is_zero_based:
                            row = row_raw  # Already 0-based
                            col = col_raw  # Already 0-based
                        else:
                            row = row_raw - 1  # Convert from 1-based
                            col = col_raw - 1  # Convert from 1-based
                        
                        # Skip invalid indices
                        if row < 0 or col < 0:
                            continue
                        value = float(parts[2]) if len(parts) >= 3 else 1.0
                        data_dict[row].append((col, value))
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
        else:
            # Simple coordinate format (no header, 0-based indexing)
            # First line might be dimensions or data
            parts = first_line.split()
            if len(parts) >= 2:
                # Check if first line is dimensions (all integers) or data
                try:
                    # Try to parse as dimensions
                    if len(parts) == 3 and '.' not in parts[0] and '.' not in parts[1]:
                        rows = int(parts[0])
                        cols = int(parts[1])
                        nnz = int(parts[2])
                        # Skip to next line
                    else:
                        # First line is data
                        row = int(parts[0])  # Already 0-based
                        col = int(parts[1])  # Already 0-based
                        value = float(parts[2]) if len(parts) >= 3 else 1.0
                        data_dict[row].append((col, value))
                        rows = max(rows, row + 1)
                        cols = max(cols, col + 1)
                except ValueError:
                    # First line is data
                    row = int(parts[0])  # Already 0-based
                    col = int(parts[1])  # Already 0-based
                    value = float(parts[2]) if len(parts) >= 3 else 1.0
                    data_dict[row].append((col, value))
                    rows = max(rows, row + 1)
                    cols = max(cols, col + 1)
            
            # Read remaining data
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        row = int(parts[0])  # Already 0-based
                        col = int(parts[1])  # Already 0-based
                        # Skip invalid indices
                        if row < 0 or col < 0:
                            continue
                        value = float(parts[2]) if len(parts) >= 3 else 1.0
                        data_dict[row].append((col, value))
                        rows = max(rows, row + 1)
                        cols = max(cols, col + 1)
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
    
    # Calculate actual nnz
    nnz = sum(len(data_dict[row]) for row in data_dict)
    
    # Sort each row by (value descending, col ascending) for consistent comparison
    for row in data_dict:
        data_dict[row].sort(key=lambda x: (-x[1], x[0]))
    
    return rows, cols, nnz, data_dict


def compare_candidate_pairs(ref_file, computed_file, topk, rel_tolerance=1e-6, abs_tolerance=1e-6):
    """
    Compare two candidate pairs MTX files.
    For each row, compare the top-k similarities (sorted by similarity value, then column index).
    
    Args:
        ref_file: Path to reference MTX file
        computed_file: Path to computed MTX file
        topk: Expected top-k value
        rel_tolerance: Relative error tolerance for similarity values
        abs_tolerance: Absolute error tolerance for similarity values
    
    Returns: (is_match, error_message, details)
    """
    print(f"\n{'='*80}")
    print(f"Comparing candidate pairs:")
    print(f"  Reference:   {ref_file}")
    print(f"  Computed:    {computed_file}")
    print(f"  Top-K:       {topk}")
    print(f"{'='*80}")
    
    # Read files
    ref_result = read_mtx_file(ref_file)
    if ref_result is None:
        return False, f"Reference file not found: {ref_file}", {}
    
    comp_result = read_mtx_file(computed_file)
    if comp_result is None:
        return False, f"Computed file not found: {computed_file}", {}
    
    ref_rows, ref_cols, ref_nnz, ref_data = ref_result
    comp_rows, comp_cols, comp_nnz, comp_data = comp_result
    
    # For candidate pairs, we only compare rows that have data
    # Dimension mismatch is acceptable if it's just about empty rows
    ref_has_data_rows = set(ref_data.keys())
    comp_has_data_rows = set(comp_data.keys())
    
    # Warn about dimension mismatch but continue comparison
    if ref_rows != comp_rows or ref_cols != comp_cols:
        print(f"  Warning: Dimension mismatch - reference={ref_rows}x{ref_cols}, computed={comp_rows}x{comp_cols}")
        print(f"  This is acceptable if only empty rows differ. Continuing comparison...")
    
    # Compare only rows that have data in at least one file
    all_rows = ref_has_data_rows | comp_has_data_rows
    differences = []
    total_rows = len(all_rows)
    matched_rows = 0
    mismatch_details = []
    
    for row in sorted(all_rows):
        ref_row_data = ref_data.get(row, [])
        comp_row_data = comp_data.get(row, [])
        
        # Check if row data matches (with tolerance for floating point values)
        # Convert to dictionaries for tolerance-based comparison
        ref_dict = {col: val for col, val in ref_row_data}
        comp_dict = {col: val for col, val in comp_row_data}
        
        # Check if all columns match with tolerance
        ref_cols_set = set(ref_dict.keys())
        comp_cols_set = set(comp_dict.keys())
        
        if ref_cols_set == comp_cols_set:
            # All columns match, check values with tolerance
            all_values_match = True
            for col in ref_cols_set:
                ref_val = ref_dict[col]
                comp_val = comp_dict[col]
                abs_diff = abs(ref_val - comp_val)
                rel_diff = abs_diff / max(abs(ref_val), abs(comp_val), 1e-15)
                
                if abs_diff > abs_tolerance and rel_diff > rel_tolerance:
                    all_values_match = False
                    break
            
            if all_values_match:
                matched_rows += 1
                continue
        
        # Detailed comparison for mismatched rows
        # (ref_dict and comp_dict already created above)
        
        # Find differences
        ref_cols_set = set(ref_dict.keys())
        comp_cols_set = set(comp_dict.keys())
        
        only_in_ref = ref_cols_set - comp_cols_set
        only_in_comp = comp_cols_set - ref_cols_set
        common_cols = ref_cols_set & comp_cols_set
        
        # Check value differences for common columns
        value_diffs = []
        for col in common_cols:
            ref_val = ref_dict[col]
            comp_val = comp_dict[col]
            abs_diff = abs(ref_val - comp_val)
            rel_diff = abs_diff / max(abs(ref_val), abs(comp_val), 1e-15)
            
            if abs_diff > abs_tolerance and rel_diff > rel_tolerance:
                value_diffs.append((col, ref_val, comp_val, abs_diff, rel_diff))
        
        if only_in_ref or only_in_comp or value_diffs:
            mismatch_details.append({
                'row': row,
                'ref_count': len(ref_row_data),
                'comp_count': len(comp_row_data),
                'only_in_ref': sorted(only_in_ref),
                'only_in_comp': sorted(only_in_comp),
                'value_diffs': value_diffs
            })
    
    # Generate summary
    details = {
        'total_rows': total_rows,
        'matched_rows': matched_rows,
        'mismatched_rows': len(mismatch_details),
        'mismatch_details': mismatch_details[:10],  # Show first 10 mismatches
        'ref_nnz': ref_nnz,
        'comp_nnz': comp_nnz
    }
    
    if len(mismatch_details) == 0:
        return True, "All rows match!", details
    else:
        error_msg = f"Mismatch in {len(mismatch_details)} out of {total_rows} rows"
        return False, error_msg, details


def print_comparison_result(matrix_name, is_match, error_message, details):
    """Print comparison result in a formatted way."""
    print(f"\n{'='*80}")
    print(f"Matrix: {matrix_name}")
    print(f"Status: {'[PASSED]' if is_match else '[FAILED]'}")
    print(f"{'='*80}")
    
    if is_match:
        print(f"[OK] All rows match perfectly!")
        print(f"  Total rows: {details['total_rows']}")
        print(f"  Total nnz:  {details['ref_nnz']}")
    else:
        print(f"[ERROR] {error_message}")
        print(f"  Matched rows:    {details['matched_rows']} / {details['total_rows']}")
        print(f"  Mismatched rows: {details['mismatched_rows']}")
        print(f"  Reference nnz:    {details['ref_nnz']}")
        print(f"  Computed nnz:    {details['comp_nnz']}")
        
        # Print first few mismatches
        if details['mismatch_details']:
            print(f"\n  First few mismatches:")
            for mismatch in details['mismatch_details'][:5]:
                row = mismatch['row']
                print(f"\n    Row {row}:")
                print(f"      Reference: {mismatch['ref_count']} elements")
                print(f"      Computed:  {mismatch['comp_count']} elements")
                
                if mismatch['only_in_ref']:
                    print(f"      Only in reference (first 5): {mismatch['only_in_ref'][:5]}")
                if mismatch['only_in_comp']:
                    print(f"      Only in computed (first 5):   {mismatch['only_in_comp'][:5]}")
                
                if mismatch['value_diffs']:
                    print(f"      Value differences (first 3):")
                    for col, ref_val, comp_val, abs_diff, rel_diff in mismatch['value_diffs'][:3]:
                        print(f"        Col {col}: ref={ref_val:.10f}, comp={comp_val:.10f}, "
                              f"diff={abs_diff:.2e}, rel_diff={rel_diff:.2e}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare candidate pairs results with reference results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 compare_candidate_pairs.py
  python3 compare_candidate_pairs.py --topk=10
  python3 compare_candidate_pairs.py --ref_dir=/path/to/ref --computed_dir=/path/to/computed
        """
    )
    
    parser.add_argument('--topk', type=int, default=DEFAULT_TOPK,
                        help=f'Top-k value used (default: {DEFAULT_TOPK})')
    parser.add_argument('--ref_dir', type=str, default=DEFAULT_REF_DIR,
                        help=f'Directory containing reference results (default: {DEFAULT_REF_DIR})')
    parser.add_argument('--computed_dir', type=str, default=DEFAULT_COMPUTED_DIR,
                        help=f'Directory containing computed results (default: {DEFAULT_COMPUTED_DIR})')
    parser.add_argument('--rel_tolerance', type=float, default=1e-6,
                        help='Relative error tolerance (default: 1e-6)')
    parser.add_argument('--abs_tolerance', type=float, default=1e-6,
                        help='Absolute error tolerance (default: 1e-6)')
    
    args = parser.parse_args()
    
    # Get script directory and resolve computed_dir relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    if not os.path.isabs(args.computed_dir):
        computed_dir = os.path.join(base_dir, args.computed_dir)
    else:
        computed_dir = args.computed_dir
    
    ref_dir = args.ref_dir
    
    print(f"{'='*80}")
    print(f"Candidate Pairs Comparison Script")
    print(f"{'='*80}")
    print(f"Reference directory:   {ref_dir}")
    print(f"Computed directory:    {computed_dir}")
    print(f"Top-K value:           {args.topk}")
    print(f"Relative tolerance:    {args.rel_tolerance}")
    print(f"Absolute tolerance:    {args.abs_tolerance}")
    print(f"{'='*80}")
    
    # Compare each matrix
    all_passed = True
    results = []
    
    for matrix_name in MATRIX_NAMES:
        # Construct file paths
        ref_file = os.path.join(ref_dir, f"{matrix_name}.mtx")
        computed_file = os.path.join(computed_dir, f"{matrix_name}_candidate_pairs_topk{args.topk}.mtx")
        
        # Check if files exist
        if not os.path.exists(ref_file):
            print(f"\n⚠ Warning: Reference file not found: {ref_file}")
            print(f"  Skipping {matrix_name}")
            continue
        
        if not os.path.exists(computed_file):
            print(f"\n⚠ Warning: Computed file not found: {computed_file}")
            print(f"  Skipping {matrix_name}")
            print(f"  Please run: ./generate_candidate_pairs {matrix_name}.mtx --topk={args.topk}")
            continue
        
        # Compare
        is_match, error_message, details = compare_candidate_pairs(
            ref_file, computed_file, args.topk,
            args.rel_tolerance, args.abs_tolerance
        )
        
        print_comparison_result(matrix_name, is_match, error_message, details)
        
        results.append((matrix_name, is_match))
        if not is_match:
            all_passed = False
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    for matrix_name, is_match in results:
        status = "[PASSED]" if is_match else "[FAILED]"
        print(f"  {matrix_name:20s} {status}")
    
    if all_passed:
        print(f"\n[SUCCESS] All comparisons passed!")
        return 0
    else:
        print(f"\n[FAILURE] Some comparisons failed. Please check the details above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
