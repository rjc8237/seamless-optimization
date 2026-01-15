import json
import numpy as np
from scipy.sparse import coo_matrix
from pathlib import Path

# Use absolute path
base_path = Path("/Users/aa13586/Desktop/symmetric-dirichlet")
# Load from JSON
folder = base_path / "output" / "filigree100k_param"
json_files = list(folder.glob("*.json"))

print(f"Found {len(json_files)} JSON files:")
for json_file in json_files:
    print(f"  - {json_file.name}")

with open(json_files[0]) as f:
    data = json.load(f)

# Debug: print the structure
print(f"\nData type: {type(data)}")
print(f"\nFirst 100 chars: {str(data)[:100]}")

# If data is a list, access first element
if isinstance(data, list):
    print(f"Data is a list with {len(data)} elements")
    hess = data[0]
else:
    hess = data["hessian"]

print(f"\nHessian type: {type(hess)}")

# hess is a list, so convert directly
triplets_array = np.array(hess, dtype=np.float64)

print(f"Triplets shape: {triplets_array.shape}")

rows = triplets_array[:, 0].astype(np.int32)
cols = triplets_array[:, 1].astype(np.int32)
vals = triplets_array[:, 2]

# You'll need to know the matrix dimensions - adjust as needed
# Or infer them from the max row/col indices
n_rows = int(rows.max()) + 1
n_cols = int(cols.max()) + 1

hessian = coo_matrix(
    (vals, (rows, cols)),
    shape=(n_rows, n_cols)
)

print(f"\nHessian shape: {hessian.shape}")
print(f"Non-zero elements: {hessian.nnz}")

# Print largest and smallest elements
print(f"\nLargest and smallest elements: {vals.max()}, {vals.min()}")
print(f"Absolute largest and smallest: {np.abs(vals).max()}, {np.abs(vals).min()}")

# Find indices of extreme values
max_idx = np.argmax(vals)
min_idx = np.argmin(vals)
abs_max_idx = np.argmax(np.abs(vals))
abs_min_idx = np.argmin(np.abs(vals))

# Get the row and column indices
max_row, max_col = rows[max_idx], cols[max_idx]
min_row, min_col = rows[min_idx], cols[min_idx]
abs_max_row, abs_max_col = rows[abs_max_idx], cols[abs_max_idx]
abs_min_row, abs_min_col = rows[abs_min_idx], cols[abs_min_idx]

# Open a file for writing
output_file = base_path / "output" / "hessian_analysis.txt"

with open(output_file, 'w') as f:
    f.write(f"Hessian shape: {hessian.shape}\n")
    f.write(f"Non-zero elements: {hessian.nnz} = {hessian.nnz / hessian.shape[0] / hessian.shape[1] * 100.0:.3f}%\n\n")

    f.write(f"Largest and smallest elements: {vals.max()}, {vals.min()}\n")
    f.write(f"Absolute largest and smallest: {np.abs(vals).max()}, {np.abs(vals).min()}\n\n")

    # Count rows with elements close to largest element
    threshold = vals.max() * 1e-9  # 0.1% of largest element
    close_to_max = np.sum(vals >= threshold)
    unique_rows_close = len(rows[vals >= threshold])
    f.write(f"Elements larger than threshold({threshold:.2e}): {close_to_max}\n")
    f.write(f"Rows containing elements larger than threshold: {unique_rows_close}\n\n")

    # Print rows containing extreme values (non-zero only)
    f.write(f"--- Row with LARGEST element ({vals[max_idx]}) ---\n")
    row_sparse = hessian.getrow(max_row)
    f.write(f"Row {max_row} - Non-zero values: {row_sparse.data}\n")

    f.write(f"--- Row with SMALLEST element ({vals[min_idx]}) ---\n")
    row_sparse = hessian.getrow(min_row)
    f.write(f"Row {min_row} - Non-zero values: {row_sparse.data}\n")

    f.write(f"--- Row with ABSOLUTE LARGEST element ({vals[abs_max_idx]}) ---\n")
    row_sparse = hessian.getrow(abs_max_row)
    f.write(f"Row {abs_max_row} - Non-zero values: {row_sparse.data}\n")

    f.write(f"--- Row with ABSOLUTE SMALLEST element ({vals[abs_min_idx]}) ---\n")
    row_sparse = hessian.getrow(abs_min_row)
    f.write(f"Row {abs_min_row} - Non-zero values: {row_sparse.data}\n")

print(f"Output written to {output_file}")