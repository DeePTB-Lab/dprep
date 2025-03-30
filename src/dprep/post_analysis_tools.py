import os
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from abacustest.constant import RY2EV
# Import all required libraries
from abacustest.lib_model.model_012_band import PostBand, ReadInput, ReadKpt
from ase.db import connect

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def copy_failed_folders(source_dir, target_folder_name, check_file_name, id_prefix, dump_dir):
    """
    Recursively finds specific folders (e.g., 'OUT.ABACUS') whose parent folder
    starts with a given prefix (e.g., 'id_'). If a specific file (e.g.,
    'BANDS_1.dat') is *missing* inside the target folder, the entire parent
    'id_' folder is copied to the dump directory.

    Args:
        source_dir (str): The root directory to start searching from.
        target_folder_name (str): The exact name of the folder to search for (e.g., 'OUT.ABACUS').
        check_file_name (str): The name of the file to check for *inside* the target_folder_name.
        id_prefix (str): The prefix of the parent directory name (e.g., 'db_seq_id_').
        dump_dir (str): The directory where qualifying parent 'id_' folders will be copied.
    """
    # --- Input Validation and Setup ---
    source_path = Path(source_dir)
    dump_path = Path(dump_dir)

    if not source_path.is_dir():
        print(f"Error: Source directory '{source_dir}' not found or is not a directory.")
        return

    if id_prefix is None:
        id_prefix = "db_seq_id_"

    # Create dump directory if it doesn't exist
    dump_path.mkdir(parents=True, exist_ok=True)

    print(f"Searching in:           {source_path}")
    print(f"Looking for folders:    {target_folder_name}")
    print(f"Checking for absence of:{check_file_name} inside {target_folder_name}")
    print(f"Identifying parent with: '{id_prefix}'")
    print(f"Copying qualifying parents to: {dump_path}")
    print("-" * 20)

    found_target_folder_count = 0
    copied_parent_count = 0

    # --- Recursive Search for the target folder name ---
    # Use rglob to find potential matches (files or folders)
    for potential_folder in source_path.rglob(target_folder_name):
        # Ensure it's actually a directory we found
        if potential_folder.is_dir() and potential_folder.name == target_folder_name:
            found_target_folder_count += 1
            print(f"Found potential folder: {potential_folder}")

            # --- Check the parent folder ---
            parent_folder = potential_folder.parent
            if parent_folder.name.startswith(id_prefix):
                print(f"  Parent '{parent_folder.name}' matches prefix '{id_prefix}'.")

                # --- Check if the specific file exists within the target folder ---
                file_to_check = potential_folder / check_file_name
                if not file_to_check.is_file():
                    print(f"  -> File '{check_file_name}' NOT found inside {potential_folder.name}.")

                    # --- Prepare to copy the parent ('id_') folder ---
                    parent_folder_to_copy = parent_folder  # This is the 'id_...' folder
                    destination_path = dump_path / parent_folder_to_copy.name

                    # --- Copy the entire parent folder ---
                    # Avoid overwriting if it already exists in the dump folder
                    if destination_path.exists():
                        print(f"  -> Skipped: Destination '{destination_path}' already exists.")
                    else:
                        try:
                            print(f"  -> Copying parent folder '{parent_folder_to_copy.name}' to '{dump_path}'...")
                            # shutil.copytree copies an entire directory tree
                            shutil.copytree(parent_folder_to_copy, destination_path)
                            copied_parent_count += 1
                            print(f"  -> Successfully copied '{parent_folder_to_copy.name}'.")
                        except Exception as e:
                            print(f"  -> Error copying {parent_folder_to_copy} to {destination_path}: {e}")
                else:
                    print(f"  -> File '{check_file_name}' found inside. Skipping copy.")
            # else:
            # Optional: Add a message if the parent doesn't match the prefix
            # print(f"  Parent '{parent_folder.name}' does not match prefix '{id_prefix}'. Skipping.")
        # else:
        # Optional: Add a message if rglob found a file with the same name
        # if potential_folder.is_file() and potential_folder.name == target_folder_name:
        #    print(f"Found item is a file, not a directory: {potential_folder}")

    print("-" * 20)
    print(f"Search complete.")
    print(f"Found {found_target_folder_count} folder(s) named '{target_folder_name}'.")
    print(
        f"Copied {copied_parent_count} parent folder(s) (matching '{id_prefix}' prefix and missing '{check_file_name}') to '{dump_path}'.")
    if copied_parent_count == 0:
        print('No errors found.')


def find_copy_rename_recursive(source_dir, filename_to_find, dest_dir, id_prefix):
    """
    Recursively finds files with a specific name within a source directory,
    copies them to a destination directory, and renames them based on a
    parent directory name starting with a given prefix.

    Args:
        source_dir (str): The root directory to start searching from.
        filename_to_find (str): The exact name of the file to search for (e.g., 'BANDS_1.dat').
        dest_dir (str): The directory where found files will be copied.
        id_prefix (str): The prefix of the directory name to use for renaming (e.g., 'id_').
    """
    # --- Input Validation and Setup ---
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.is_dir():
        print(f"Error: Source directory '{source_dir}' not found or is not a directory.")
        return

    # Create destination directory if it doesn't exist
    # The exist_ok=True prevents an error if the directory already exists.
    # The parents=True creates any necessary parent directories as well.
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"Searching in: {source_path}")
    print(f"Looking for:  {filename_to_find}")
    print(f"Copying to:   {dest_path}")
    print(f"ID prefix:    '{id_prefix}'")
    print("-" * 20)

    found_count = 0
    copied_count = 0

    # --- Recursive Search using rglob ---
    # Path.rglob performs a recursive search for patterns
    for found_file in source_path.rglob(filename_to_find):
        found_count += 1
        print(f"Found: {found_file}")

        # --- Extract ID for Renaming ---
        id_name = None
        # Iterate through the parent directories of the found file
        # Use found_file.parents which gives a sequence from immediate parent upwards
        for parent in found_file.parents:
            if parent.name.startswith(id_prefix):
                id_name = parent.name
                break  # Stop once the first matching parent is found
            if parent == source_path:  # Stop if we reach the source search directory
                break

        if id_name:
            # --- Construct New Name and Copy ---
            # Get the original file's extension (e.g., '.dat')
            original_extension = found_file.suffix
            new_filename = f"{id_name}{original_extension}"
            destination_file_path = dest_path / new_filename

            try:
                print(f"  -> Copying to: {destination_file_path}")
                # shutil.copy2 attempts to preserve metadata like modification time
                shutil.copy2(found_file, destination_file_path)
                copied_count += 1
            except Exception as e:
                print(f"  -> Error copying {found_file} to {destination_file_path}: {e}")
        else:
            print(
                f"  -> Warning: Could not find parent directory starting with '{id_prefix}' for {found_file}. Skipping rename/copy.")

    print("-" * 20)
    print(f"Search complete. Found {found_count} file(s). Copied and renamed {copied_count} file(s).")


# --- Helper function (Internal) ---
def _extract_single_job_data(job_dir_path: Path, target_band_file: str):
    """
    Internal helper: Extracts band data for a single job directory.
    Mimics logic from the original PostBand class for finding files and data.

    Args:
        job_dir_path (Path): Path to the ID directory (e.g., .../pw/id_1).
        target_band_file (str): Name of the band file (e.g., BANDS_1.dat).

    Returns:
        tuple: (bands, kpt_lines, efermi, error_message)
    """
    try:
        job_dir = str(job_dir_path)
        post_instance = PostBand([job_dir])

        # Find INPUT file
        inputfile_path = None
        for ifile in ["INPUT.nscf", "INPUT"]:
            potential_path = job_dir_path / ifile
            if potential_path.is_file():
                inputfile_path = potential_path
                break
        if not inputfile_path:
            return None, None, None, f"Input file not found in {job_dir}"

        input_param = ReadInput(str(inputfile_path))

        # Get KPT data
        kptfile_path = None
        kpt_file_rel = input_param.get("kpoint_file")
        if kpt_file_rel:
            kptfile_path = job_dir_path / kpt_file_rel

        if not kptfile_path or not kptfile_path.is_file():
            # Try common alternatives if specific one not found or specified
            for kfile in ["KPT.nscf", "KPT", "KLINES", "LINES"]:
                potential_path = job_dir_path / kfile
                if potential_path.is_file():
                    kptfile_path = potential_path
                    break
        if not kptfile_path or not kptfile_path.is_file():
            return None, None, None, f"KPT file not found in {job_dir}"

        kpt_data = ReadKpt(str(kptfile_path))
        if kpt_data is None or kpt_data[1] != 'line':
            return None, None, None, f"KPT file '{kptfile_path.name}' not found or not line mode."
        kpt_lines = kpt_data[0]

        # Find band and log files
        suffix = input_param.get("suffix", "ABACUS")
        out_dir = job_dir_path / f"OUT.{suffix}"
        band_file = out_dir / target_band_file
        log_file = out_dir / "running_nscf.log"

        if not band_file.is_file():
            return None, None, None, f"Band file '{band_file.name}' not found in {out_dir.name}."
        if not log_file.is_file():
            log_file = out_dir / "running_scf.log"  # Fallback
            if not log_file.is_file():
                return None, None, None, f"Log file (*nscf.log / *scf.log) not found in {out_dir.name}."
            else:
                print(f"  Warning: Using scf log for Fermi energy in {job_dir_path.name}")

        # Get Bands and E_Fermi
        bands = PostBand.get_band(str(band_file))
        efermi = post_instance.get_efermi(str(log_file))

        if efermi is None:
            return None, None, None, f"Could not extract E_Fermi from '{log_file.name}'."
        if bands is None or bands.size == 0:
            return None, None, None, f"Could not extract bands from '{band_file.name}'."

        return bands, kpt_lines, efermi, None

    except Exception as e:
        return None, None, None, f"Error processing {job_dir_path.name}: {type(e).__name__}"


# --- Core Functions ---
def process_band_data(
        workspace_root: str,
        plot_data_dir: str,
        job_types: list = ['pw', 'lcao'],
        id_prefix: str = 'id_',
        target_folder: str = 'OUT.ABACUS',
        target_band_file: str = 'BANDS_1.dat',
        force_reprocess: bool = False
):
    """
    Finds PW/LCAO band data, extracts relevant information, and saves it
    to .npz files for later plotting, avoiding reprocessing if data exists.

    Args:
        workspace_root (str): Path to the main comparison directory.
        plot_data_dir (str): Path to the directory where processed .npz files will be stored.
        job_types (list): List of job type subfolder names (e.g., ['pw', 'lcao']).
        id_prefix (str): Prefix used for the ID directories.
        target_folder (str): Name of the output folder within ID directories.
        target_band_file (str): Name of the band data file.
        force_reprocess (bool): If True, overwrite existing .npz files.

    Returns:
        dict: A dictionary containing lists of IDs.
    """
    workspace_path = Path(workspace_root)
    plot_data_path = Path(plot_data_dir)
    plot_data_path.mkdir(parents=True, exist_ok=True)

    print("\n--- Starting Data Processing ---")
    print(f"Workspace: {workspace_path}")
    print(f"Job Types: {job_types}")
    print(f"ID Prefix: '{id_prefix}'")
    print(f"Target:    '{target_folder}/{target_band_file}'")
    print(f"Cache Dir: {plot_data_path}")
    print(f"Force Reprocess: {force_reprocess}")
    print("-" * 30)

    # 1. Find all potential jobs
    found_jobs = defaultdict(lambda: {job_type: None for job_type in job_types})
    raw_band_files_count = 0

    for job_type in job_types:
        search_dir = workspace_path / job_type
        if not search_dir.is_dir():
            print(f"Warning: Search directory not found: {search_dir}")
            continue

        for band_file_path in search_dir.rglob(f"**/{target_folder}/{target_band_file}"):
            raw_band_files_count += 1
            out_folder = band_file_path.parent
            id_dir = out_folder.parent
            if id_dir.name.startswith(id_prefix):
                id_name = id_dir.name
                if job_type in found_jobs[id_name]:
                    found_jobs[id_name][job_type] = id_dir

    print(f"Found {raw_band_files_count} raw band files across specified job types.")

    # 2. Process pairs and save data
    processed_ids = []
    missing_pair_ids = []
    error_ids = defaultdict(list)
    skipped_ids = []

    sorted_ids = sorted(found_jobs.keys())

    for id_name in sorted_ids:
        job_paths = found_jobs[id_name]

        if not all(job_paths[jt] for jt in job_types):
            missing_pair_ids.append(id_name)
            continue

        output_npz_path = plot_data_path / f"{id_name}.npz"

        if output_npz_path.exists() and not force_reprocess:
            skipped_ids.append(id_name)
            continue

        print(f"  Processing {id_name}...")
        data_to_save = {}
        has_error = False

        for i, job_type in enumerate(job_types):
            job_dir = job_paths[job_type]
            bands, kpt_lines, efermi, err = _extract_single_job_data(job_dir, target_band_file)

            if err:
                print(f"    Error ({job_type}): {err}")
                error_ids[id_name].append(f"{job_type}: {err}")
                has_error = True
                break

            data_to_save[f'bands_{job_type}'] = bands
            data_to_save[f'efermi_{job_type}'] = efermi
            if i == 0:
                data_to_save['kpt_lines_ref'] = np.array(kpt_lines, dtype=object)

        if not has_error:
            try:
                np.savez_compressed(output_npz_path, **data_to_save)
                processed_ids.append(id_name)
            except Exception as e:
                print(f"    Error saving .npz for {id_name}: {e}")
                error_ids[id_name].append(f"Saving NPZ Error: {e}")
                if output_npz_path.exists():
                    try:
                        output_npz_path.unlink()
                    except OSError:
                        print(f"    Warning: Could not remove partially written file {output_npz_path.name}")

    print("-" * 30)
    print("Processing Summary:")
    print(f"- IDs processed and saved: {len(processed_ids)}")
    print(f"- IDs skipped (already processed): {len(skipped_ids)}")
    print(f"- IDs missing a required job type: {len(missing_pair_ids)}")
    print(f"- IDs with errors during data extraction/saving: {len(error_ids)}")
    if error_ids:
        print("  Error Details:")
        for id_name, errors in error_ids.items():
            print(f"    - {id_name}: {'; '.join(errors)}")
    print("-" * 30)

    return {
        'processed': processed_ids,
        'missing_pair': missing_pair_ids,
        'errors': list(error_ids.keys()),
        'skipped': skipped_ids
    }


def plot_band_comparisons(
        plot_data_dir: str,
        pics_dir: str,
        job_types: list = ['pw', 'lcao'],
        id_prefix: str = 'id_',
        ase_db_path: str = None,
        plot_styles: dict = {'pw': {'color': 'blue', 'linestyle': '-', 'label': 'PW'},
                             'lcao': {'color': 'red', 'linestyle': '--', 'label': 'LCAO'}},
        plot_ylim: list = [-5, 5],
        plot_filename_suffix: str = '_compare.png',
        force_replot: bool = False
):
    """
    Generates comparison band structure plots from cached .npz data.
    Optionally uses an ASE database to add chemical formulas to titles.

    Args:
        plot_data_dir (str): Directory containing the processed .npz files.
        pics_dir (str): Directory where output plot images will be saved.
        job_types (list): List of job types used during processing.
        id_prefix (str): Prefix used for the ID directories.
        ase_db_path (str, optional): Path to the ASE database.
        plot_styles (dict): Dictionary mapping job_type to plot style parameters.
        plot_ylim (list): Y-axis limits relative to Fermi level [min, max].
        plot_filename_suffix (str): Suffix for the output plot filenames.
        force_replot (bool): If True, overwrite existing plot files.
    """
    plot_data_path = Path(plot_data_dir)
    pics_path = Path(pics_dir)
    pics_path.mkdir(parents=True, exist_ok=True)

    if not plot_data_path.is_dir():
        print(f"Error: Plot data directory not found: {plot_data_path}")
        return

    # --- ASE DB Setup ---
    db_conn = None
    if ase_db_path:
        ase_db_file = Path(ase_db_path)
        if not ase_db_file.is_file():
            print(f"Warning: ASE DB file not found at '{ase_db_path}'. Cannot fetch formulas.")
        else:
            try:
                db_conn = connect(ase_db_file)
                print(f"Connected to ASE DB: {ase_db_file.name}")
            except Exception as e:
                print(f"Warning: Failed to connect to ASE DB '{ase_db_path}': {e}")
                db_conn = None

    print("\n--- Starting Plot Generation ---")
    print(f"Reading data from: {plot_data_path}")
    print(f"Saving plots to:   {pics_path}")
    print(f"Plot Y limits:     {plot_ylim}")
    print(f"ASE DB for titles: {'Yes (' + Path(ase_db_path).name + ')' if db_conn else 'No'}")
    print(f"Force Replot:      {force_replot}")
    print("-" * 30)

    plot_count = 0
    plot_errors = 0
    skipped_count = 0
    npz_files = sorted(list(plot_data_path.glob(f"{id_prefix}*.npz")))

    if not npz_files:
        print("No processed .npz files matching the ID prefix found to plot.")
        print("-" * 30)
        return

    for npz_file in npz_files:
        id_name = npz_file.stem
        print(f"  Plotting {id_name}...")

        try:
            data = np.load(npz_file, allow_pickle=True)

            required_keys = ['kpt_lines_ref']
            for jt in job_types:
                required_keys.extend([f'bands_{jt}', f'efermi_{jt}'])
            if not all(key in data for key in required_keys):
                print(f"    Error: Missing required data keys in {npz_file.name}. Skipping.")
                plot_errors += 1
                continue

            kpt_lines_ref = data['kpt_lines_ref']
            bands_ref = data[f'bands_{job_types[0]}']
            band_idx, symbol_index, symbols = PostBand.rearrange_plotdata(bands_ref, kpt_lines_ref)

            # --- Get Title ID (Formula or ID Name) ---
            plot_title_id = id_name  # Default to id_name

            if db_conn:
                try:
                    numeric_id_str = id_name.replace(id_prefix, "")
                    if numeric_id_str.isdigit():
                        original_id_zero_based = int(numeric_id_str)
                        ase_id_one_based = original_id_zero_based + 1
                        row = db_conn.get(id=ase_id_one_based)
                        if row and hasattr(row, 'formula'):
                            plot_title_id = row.formula
                except Exception as db_err:
                    print(f"    Warning: Error querying ASE DB for {id_name}: {db_err}")

            plot_filename = pics_path / (plot_title_id + plot_filename_suffix)

            if plot_filename.exists() and not force_replot:
                skipped_count += 1
                continue

            fig, ax = plt.subplots(figsize=(6, 5))

            for job_type in job_types:
                bands = data[f'bands_{job_type}']
                efermi = data[f'efermi_{job_type}']
                style = plot_styles.get(job_type, {})

                if bands.shape[1] != bands_ref.shape[1]:
                    print(
                        f"    Warning: Band dimension mismatch for {job_type} in {id_name}. Skipping {job_type} plot.")
                    continue

                for band_num, iband in enumerate(bands):
                    label = style.get('label', job_type) if band_num == 0 else ""
                    color = style.get('color', 'black')
                    ls = style.get('linestyle', '-')
                    for i, idx in enumerate(band_idx):
                        ax.plot(range(idx[0], idx[1]), iband[idx[2]:idx[3]] - efermi,
                                linestyle=ls, color=color, linewidth=1.0, alpha=0.8, label=label if i == 0 else "")

            ax.set_xlim(0, band_idx[-1][1])
            ax.set_ylim(plot_ylim[0], plot_ylim[1])
            if symbols is not None and symbol_index is not None:
                ax.set_xticks(symbol_index)
                ax.set_xticklabels(symbols)
                for index in symbol_index:
                    ax.axvline(x=index, color='gray', linestyle=':', linewidth=0.6)

            ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel("K points")
            ax.set_ylabel("Energy (E - E$_f$, eV)")
            ax.set_title(f"Band Comparison: {plot_title_id}")
            ax.legend()

            plt.tight_layout()
            plt.savefig(plot_filename, dpi=600)
            plt.close(fig)
            plot_count += 1

        except Exception as e:
            print(f"    Error plotting {id_name}: {type(e).__name__} - {e}")
            plot_errors += 1
            plt.close('all')

    print("-" * 30)
    print("Plotting Summary:")
    print(f"- Plots generated: {plot_count}")
    print(f"- Plots skipped (already exist): {skipped_count}")
    print(f"- Errors during plotting: {plot_errors}")
    print("-" * 30)


# --- Main Execution Example ---
def run_band_comparison_workflow(
        workspace_root: str,
        base_output_dir: str,
        ase_db_path: str = None,
        job_types: list = ['pw', 'lcao'],
        id_prefix: str = 'id_',
        target_folder: str = 'OUT.ABACUS',
        target_band_file: str = 'BANDS_1.dat',
        plot_styles: dict = {'pw': {'color': 'blue', 'linestyle': '-', 'label': 'PW'},
                             'lcao': {'color': 'red', 'linestyle': '--', 'label': 'LCAO'}},
        plot_ylim: list = [-5, 5],
        force_reprocess: bool = False,
        force_replot: bool = False
):
    """
    Main wrapper function to run the band data processing and plotting workflow.

    Args:
        workspace_root (str): Root directory containing job_type subfolders.
        base_output_dir (str): Directory where 'plot_data' and 'pics' subdirs will be created.
        ase_db_path (str, optional): Path to the ASE database for enriching plot titles.
        job_types (list): List of job type names.
        id_prefix (str): Prefix for ID directories.
        target_folder (str): Folder name containing band results.
        target_band_file (str): Band data filename.
        plot_styles (dict): Plotting styles per job type.
        plot_ylim (list): Y-axis limits for plots.
        force_reprocess (bool): Overwrite existing processed data (.npz).
        force_replot (bool): Overwrite existing plot images (.png).
    """
    workspace_path = Path(workspace_root)
    base_output_path = Path(base_output_dir)

    if not base_output_path.is_absolute():
        base_output_path = workspace_path / base_output_dir

    plot_data_dir = base_output_path / 'plot_data'
    pics_dir = base_output_path / 'pics'

    print("=" * 40)
    print("Starting Band Comparison Workflow")
    print("=" * 40)

    # Step 1: Process data (find, extract, save to cache)
    process_results = process_band_data(
        workspace_root=str(workspace_path),
        plot_data_dir=str(plot_data_dir),
        job_types=job_types,
        id_prefix=id_prefix,
        target_folder=target_folder,
        target_band_file=target_band_file,
        force_reprocess=force_reprocess
    )

    # Step 2: Generate plots from cached data
    plot_band_comparisons(
        plot_data_dir=str(plot_data_dir),
        pics_dir=str(pics_dir),
        job_types=job_types,
        id_prefix=id_prefix,
        ase_db_path=ase_db_path,
        plot_styles=plot_styles,
        plot_ylim=plot_ylim,
        force_replot=force_replot
    )

    print("\nWorkflow Complete.")
    print("=" * 40)


if __name__ == "__main__":
    # --- Configure and Run ---
    run_band_comparison_workflow(
        workspace_root='/root/compare_band_mp_20',
        base_output_dir='band_data',
        # --- ASE Database for Titles ---
        ase_db_path='/root/compare_band_mp_20/lcao/elementary_substances.db',
        # --- Job Types and IDs ---
        job_types=['pw', 'lcao'],
        id_prefix='db_seq_id_',
        # --- Plotting Options ---
        plot_ylim=[-10, 10],
        plot_styles={'pw': {'color': '#1f77b4', 'linestyle': '-', 'label': 'PW'},
                     'lcao': {'color': '#ff7f0e', 'linestyle': '--', 'label': 'LCAO'}},
        # --- Caching Control ---
        force_reprocess=False,
        force_replot=False
    )
