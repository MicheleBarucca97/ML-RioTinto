import h5py
import numpy as np

# Point this to your master dataset
H5_PATH = "../ml_alu_data/master_dataset.h5"
GRID_PATH = "fields_midacd/nodes"

def debug_grid_mismatch():
    with h5py.File(H5_PATH, "r") as f:
        # Get all runs that have the required grid
        runs = [k for k in f.keys() if GRID_PATH in f[k]]
        if len(runs) < 2:
            print("Not enough runs to compare.")
            return
            
        ref_name = runs[0]
        ref_grid = f[ref_name][GRID_PATH][:]
        
        print(f"Reference Run : {ref_name}")
        print(f"Grid Shape    : {ref_grid.shape}")
        
        for name in runs[1:10]: # Check the first few runs
            other_grid = f[name][GRID_PATH][:]
            
            # 1. Check for Shape Mismatch (e.g., node count differs)
            if other_grid.shape != ref_grid.shape:
                print(f"\n[!] SHAPE MISMATCH in {name}")
                print(f"    Expected {ref_grid.shape}, got {other_grid.shape}")
                continue
                
            # 2. Check maximum absolute difference per axis
            diff = np.abs(other_grid - ref_grid)
            max_diff_x = np.max(diff[:, 0])
            max_diff_y = np.max(diff[:, 1])
            max_diff_z = np.max(diff[:, 2]) if diff.shape[1] > 2 else 0.0
            
            max_total_diff = np.max(diff)
            
            if max_total_diff > 1e-6:
                print(f"\n[!] GRID MISMATCH in {name}")
                print(f"    Max Diff X : {max_diff_x:.8e}")
                print(f"    Max Diff Y : {max_diff_y:.8e}")
                print(f"    Max Diff Z : {max_diff_z:.8e}")
                
                # Check if it's just a node ordering issue
                ref_sorted = np.sort(ref_grid, axis=0)
                other_sorted = np.sort(other_grid, axis=0)
                sorted_diff = np.max(np.abs(other_sorted - ref_sorted))
                
                if sorted_diff < 1e-6:
                    print("    Diagnosis  : Nodes are identical, but ORDERING is different.")
                else:
                    print("    Diagnosis  : Nodes have physically moved or floating-point noise is too high.")
                    
                return # Stop after finding the first detailed error

debug_grid_mismatch()