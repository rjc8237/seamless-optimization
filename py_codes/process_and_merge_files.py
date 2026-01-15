import pandas as pd
import re
import os
from pathlib import Path

def process_and_merge_files(file_paths, output_filename='stopping_threshold_comparisons.csv', output_dir=None):
    """
    Loads multiple comparison CSV files, reshapes them to split CG and Ch_LLT rows,
    extracts n and m parameters from filenames, and merges them into a single DataFrame.
    
    Args:
        file_paths (list): List of strings pointing to the CSV files.
        output_filename (str): Name of the output CSV file to save.
        
    Returns:
        pd.DataFrame: The merged and processed DataFrame.
    """

    # Handle output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
    else:
        output_path = output_filename

    merged_frames = []

    # Regex to extract parameters from filename (e.g., comparison_p=1_err=0.01.csv)
    # Assuming p maps to 'n' and err maps to 'm' based on file structure provided.
    filename_pattern = re.compile(r'p=([\d.]+?)_err=([\d.]+?)\.csv')

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        
        # Extract n (p) and m (err) from filename
        match = filename_pattern.search(file_name)
        if match:
            p = float(match.group(1))
            err = float(match.group(2))
        else:
            print(f"Warning: Could not extract p and err from filename: {file_name}. Setting to NaN.")
            p = float('nan')
            err = float('nan')

        # Load CSV, discarding the first two rows (header is on the 3rd row, index 2)
        try:
            df = pd.read_csv(file_path, header=1)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue

        # Identify base columns (Mesh, faces) and split remaining columns by suffix
        base_cols = ['Mesh', 'faces']
        
        # Filter columns for CG and Ch_LLT
        cg_cols = [c for c in df.columns if c.endswith('_CG')]
        ch_cols = [c for c in df.columns if c.endswith('_Ch_LLT')]
        
        # --- Process CG Rows ---
        df_cg = df[base_cols + cg_cols].copy()
        
        # Rename columns: remove '_CG' suffix
        rename_map_cg = {col: col[:-3] for col in cg_cols}
        # Handle specific naming fix if needed (e.g. converge_reason usually doesn't need extra help if suffix removal works)
        df_cg.rename(columns=rename_map_cg, inplace=True)
        
        df_cg['solver_type'] = 'CG'
        df_cg['p'] = p
        df_cg['err'] = err

        # --- Process Ch_LLT Rows ---
        df_ch = df[base_cols + ch_cols].copy()
        
        # Rename columns: remove '_Ch_LLT' suffix
        rename_map_ch = {col: col[:-7] for col in ch_cols}
        df_ch.rename(columns=rename_map_ch, inplace=True)
        
        df_ch['solver_type'] = 'Ch_LLT'
        df_ch['p'] = p
        df_ch['err'] = err

        # Align columns explicitly to ensure concatenation works smoothly 
        # (Though rename logic usually makes them identical)
        
        # Combine the two splits for this file
        df_file_processed = pd.concat([df_cg, df_ch], ignore_index=True)
        merged_frames.append(df_file_processed)

    # Concatenate all processed files
    if merged_frames:
        final_df = pd.concat(merged_frames, ignore_index=True)
        
        # Reorder columns to match requested format:
        # Mesh faces  solver_type  n  m   total_time  E_worst  convergence_reason ...
        
        # Identify variable columns (metrics like E>=0.5, etc.)
        # We exclude the fixed known columns to find the dynamic ones
        fixed_cols = ['Mesh', 'faces', 'solver_type', 'n', 'm', 'total_time', 'E_worst', 'converge_reason']
        
        # Note: In source, column is likely 'converge_reason', but rename logic might yield 'converge_reason' 
        # checking the source file snippet: 'converge_reason_CG' -> remove 3 chars -> 'converge_reason'
        
        # Get remaining columns (the E>=x ones) sorted naturally if possible
        metric_cols = [c for c in final_df.columns if c not in fixed_cols]
        
        # It's good practice to ensure 'converge_reason' column name is consistent. 
        # The script renaming logic handles it naturally based on input.
        
        final_column_order = fixed_cols + metric_cols
        
        # Ensure all columns exist (handle potential missing cols gracefully)
        existing_cols = [c for c in final_column_order if c in final_df.columns]
        final_df = final_df[existing_cols]

        # Save to CSV
        final_df.to_csv(output_path, index=False)
        print(f"Successfully created {output_path} with {len(final_df)} rows.")
        return final_df
    else:
        print("No data found.")
        return pd.DataFrame()