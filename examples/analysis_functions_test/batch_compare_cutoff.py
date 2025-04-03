from pathlib import Path
import os
from dprep.post_analysis_tools import compare_multiple_job_pairs_workflow

# Example usage block:
if __name__ == '__main__':
    # --- User Configuration ---
    WORK_ROOT = Path("./")  # Dir containing subdirs like ./lcao/, ./pw/, etc.
    OUTPUT_DIR = Path("analysis_results")
    ASE_DB_PATH = "./PW/elementary_substances.db"  # Or None
    ASE_DB_PATH = os.path.abspath(ASE_DB_PATH)

    # List of job type identifiers (must match subdirectory names)
    ALL_JOB_TYPES = ['6au', '7au', '10au']
    # Specify which job type serves as the reference
    REFERENCE_JOB_TYPE = '10au'

    # # List of job type identifiers (must match subdirectory names)
    # ALL_JOB_TYPES = ['3s2p1d', '4s3p2d1f', '5s4p3d2f', 'PW']
    # # Specify which job type serves as the reference
    # REFERENCE_JOB_TYPE = 'PW'

    # --- Run Workflow ---
    if not WORK_ROOT.is_dir():
        print(f"Error: Workspace directory not found: {WORK_ROOT}")
    else:
        if ASE_DB_PATH and not Path(ASE_DB_PATH).is_file():
            print(f"Warning: Specified ASE DB file not found: {ASE_DB_PATH}. Proceeding without it.")
            ASE_DB_PATH = None

        compare_multiple_job_pairs_workflow(
            workspace_root=str(WORK_ROOT),
            base_output_dir=str(OUTPUT_DIR),
            job_types=ALL_JOB_TYPES,
            ref_job_type=REFERENCE_JOB_TYPE,
            ase_db_path=ASE_DB_PATH,
            id_prefix='db_seq_id_',  # Adjust if your ID prefix is different
            plot_ylim=[-10, 10],
            # force_reprocess=True,
            # force_replot=True
        )