# --- Imports ---
from pathlib import Path
import numpy as np
from dprep.post_analysis_tools import _extract_single_job_data, quantify_band_error


# Assume the functions _extract_single_job_data, quantify_band_error, BandErrorMetrics are defined above
# Or import them if they are in a separate file

# --- Paths to your calculations ---
calc_dir_6au = Path('./6au/db_seq_id_17')
calc_dir_10au = Path('./10au/db_seq_id_17')
target_band_file = 'BANDS_1.dat'  # Or the specific name if different

# --- Extract data ---
print("Extracting data for 10au (Reference)...")
bands_10au, kpt_lines_10au, efermi_10au, err_10au = _extract_single_job_data(calc_dir_10au, target_band_file)

print("Extracting data for 6au (Comparison)...")
bands_6au, kpt_lines_6au, efermi_6au, err_6au = _extract_single_job_data(calc_dir_6au, target_band_file)

# --- Check for errors ---
if err_10au:
    print(f"Error loading 10au data: {err_10au}")
if err_6au:
    print(f"Error loading 6au data: {err_6au}")

# --- Quantify errors (if data loaded successfully) ---
if bands_10au is not None and bands_6au is not None and efermi_10au is not None and efermi_6au is not None:
    print("\nQuantifying errors (10au vs 6au)...")
    # Use 10au as reference (bands1, efermi1)
    error_results = quantify_band_error(bands_10au, efermi_10au, bands_6au, efermi_6au)

    if error_results:
        print("\n--- Error Quantification Results (10au vs 6au) ---")
        print(f"Reference VBM Index (10au): {error_results['vbm_index_ref']}")

        occ_metrics = error_results['occupied']
        print("\nOccupied Bands:")
        print(f"  MAE:   {occ_metrics.mae:.6f} eV")

        occ_metrics = error_results['un_occupied']
        print("\nun Occupied Bands:")
        print(f"  MAE:   {occ_metrics.mae:.6f} eV")

        occ_metrics = error_results['all_band']
        print("\nAll Bands:")
        print(f"  MAE:   {occ_metrics.mae:.6f} eV")

        val_metrics = error_results['near_ef']
        print("\nNear Ef Bands:")
        print(f"  MAE:   {val_metrics.mae:.6f} eV")

        val_metrics = error_results['valence_near_ef']
        print("\nValence Bands near Ef:")
        print(f"  MAE:   {val_metrics.mae:.6f} eV")

        cond_metrics = error_results['conduction_near_ef']
        print("\nConduction Bands near Ef:")
        print(f"  MAE:   {cond_metrics.mae:.6f} eV")
        print("-" * 50)
    else:
        print("\nError quantification failed.")
else:
    print("\nCould not quantify errors due to data loading issues.")