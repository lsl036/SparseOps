#!/usr/bin/env python3
"""
Row-wise SpGEMM test script.
Runs test_spgemm on multiple datasets with specified kernel.

Usage:
    python3 run_rowwise_spgemm.py [--kernel=1|2|3] [--test_type=correctness|performance] [--iterations=N]
    
    --kernel: Select kernel to test (default: 1)
              1: Hash-based row-wise SpGEMM
              2: Array-based row-wise SpGEMM (original)
              3: Optimized array-based row-wise SpGEMM
    
    --test_type: Select test type (default: correctness)
                 correctness: Test correctness and save results
                 performance: Benchmark performance
    
    --iterations: Number of iterations for performance test (default: 10)
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Configuration
SPARSEOPS_ROOT = "/data/lsl/SparseOps"
DATA_PATH = "/data/suitesparse_collection"

# Datasets to test
DATASETS = [
    "bcspwr10",
    "bcsstk32",
    "skirt_id_764"
]

# Kernel names for display
KERNEL_NAMES = {
    1: "Hash-based row-wise",
    2: "Array-based row-wise (original)",
    3: "Optimized array-based row-wise"
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run row-wise SpGEMM tests on multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_rowwise_spgemm.py --kernel=1
  python3 run_rowwise_spgemm.py --kernel=2 --test_type=performance
  python3 run_rowwise_spgemm.py --kernel=3 --test_type=correctness
  python3 run_rowwise_spgemm.py --kernel=1 --iterations=20
        """
    )
    
    parser.add_argument(
        "--kernel",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Kernel to test: 1 (Hash-based), 2 (Array-based original), 3 (Array-based optimized, default: 1)"
    )
    
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["correctness", "performance"],
        default="correctness",
        help="Test type: correctness or performance (default: correctness)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for performance test (default: 10)"
    )
    
    return parser.parse_args()


def setup_environment():
    """Set up environment variables for OpenMP."""
    os.environ["OMP_PROC_BIND"] = "spread"
    os.environ["OMP_PLACES"] = "cores"


def run_test(dataset, kernel, test_type, iterations):
    """Run test_spgemm for a single dataset."""
    bin_path = os.path.join(SPARSEOPS_ROOT, "build", "test_spgemm")
    mat_path = os.path.join(DATA_PATH, dataset, f"{dataset}.mtx")
    
    # Check if binary exists
    if not os.path.exists(bin_path):
        print(f"ERROR: Binary not found: {bin_path}")
        print("       Please build the project first: cd build && cmake .. && make")
        return False
    
    # Check if matrix file exists
    if not os.path.exists(mat_path):
        print(f"ERROR: Matrix file not found: {mat_path}")
        return False
    
    # Build command
    cmd = [
        bin_path,
        mat_path,
        mat_path,  # A * A (self-multiplication)
        f"--kernel={kernel}",
        f"--test_type={test_type}"
    ]
    
    # Add iterations for performance test
    if test_type == "performance":
        cmd.append(f"--iterations={iterations}")
    
    # Print header
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Kernel: {kernel} ({KERNEL_NAMES[kernel]})")
    print(f"Test Type: {test_type}")
    if test_type == "performance":
        print(f"Iterations: {iterations}")
    print("=" * 60)
    
    # Run command in build directory so output files are written there
    build_dir = os.path.join(SPARSEOPS_ROOT, "build")
    if not os.path.exists(build_dir):
        os.makedirs(build_dir, exist_ok=True)
    
    # Run command
    try:
        result = subprocess.run(
            cmd,
            check=False,  # Don't raise exception on non-zero exit
            cwd=build_dir  # Run in build directory so output files are written there
        )
        
        if result.returncode == 0:
            print(f"[SUCCESS] {dataset} completed successfully\n")
            return True
        else:
            print(f"[FAILED] {dataset} exited with code {result.returncode}\n")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to run test for {dataset}: {e}\n")
        return False


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    
    # Print configuration
    print("=" * 60)
    print("Row-wise SpGEMM Test Script")
    print("=" * 60)
    print(f"Kernel: {args.kernel} ({KERNEL_NAMES[args.kernel]})")
    print(f"Test Type: {args.test_type}")
    if args.test_type == "performance":
        print(f"Iterations: {args.iterations}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Total datasets: {len(DATASETS)}")
    print("=" * 60)
    print()
    
    # Run tests on all datasets
    results = []
    for dataset in DATASETS:
        success = run_test(dataset, args.kernel, args.test_type, args.iterations)
        results.append((dataset, success))
    
    # Print summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    for dataset, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {dataset}: {status}")
    
    print()
    print(f"Total: {len(results)} datasets")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)
    
    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
