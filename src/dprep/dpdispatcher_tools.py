import logging
import math
import os
import random
import shutil
import subprocess
import time
from datetime import datetime
from queue import Queue
from threading import Thread
import numpy as np

from ase.db.core import connect
from ase.io.abacus import write_abacus
from dpdispatcher import Task, Submission, Machine, Resources
from dprep.get_pp_orb_info import generate_pp_orb_dict
from dprep.post_analysis_tools import copy_failed_folders, save_direct_kpoints
# 新增 import
from dptb.postprocess.write_abacus_csr_file import write_blocks_to_abacus_csr

# Configure logging to file 'job_monitor.log'
logging.basicConfig(filename='job_monitor.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==========================================
# Helper Functions
# ==========================================

def get_abacus_csr_name(matrix_symbol: str) -> str:
    """根据 matrix_symbol 返回标准的 Abacus CSR 文件名。"""
    symbol_map = {
        'DM': 'dmrs1_nao.csr',
        'H': 'hrs1_nao.csr',
        'S': 'srs1_nao.csr'
    }
    return symbol_map.get(matrix_symbol, f'{matrix_symbol.lower()}rs1_nao.csr')


# ==========================================
# Main Logic
# ==========================================

def submit_job(job_folder, cmd_line):
    """Submits a job to the local system using subprocess."""
    os.chdir(job_folder)
    process = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return process


def log_remaining_time(start_time, n_completed_jobs, n_total_jobs):
    if n_completed_jobs == 0:
        logging.info(f"Waiting for the first batch to finish, connot estimate the time right now.")
        return
    elapsed_time = (datetime.now() - start_time).total_seconds()
    remaining_jobs = n_total_jobs - n_completed_jobs
    avg_time_per_job = elapsed_time / n_completed_jobs
    estimated_remaining_time = avg_time_per_job * remaining_jobs
    logging.info(f"Completed {n_completed_jobs}/{n_total_jobs} jobs. "
                 f"Estimated remaining time: {estimated_remaining_time / 3600:.2f} hours.")


def clean_out_files(out_folder_path, rm_out_files_list=['INPUT']):
    # Using topdown=True allows us to remove directories during traversal
    # and avoid descending into directories that have been deleted.
    for dirpath, dirnames, filenames in os.walk(out_folder_path, topdown=True):
        # Process directories: iterate over a copy of dirnames.
        for dirname in list(dirnames):
            if dirname in rm_out_files_list:
                full_dir_path = os.path.join(dirpath, dirname)
                shutil.rmtree(full_dir_path)
                # Remove the directory name from the list to avoid walking into it.
                dirnames.remove(dirname)
        # Process files.
        for filename in filenames:
            if filename in rm_out_files_list:
                full_file_path = os.path.join(dirpath, filename)
                os.remove(full_file_path)


def monitor_jobs(job_queue, active_jobs, cmd_line, clean_files_flag, rm_out_files_list):
    """Monitors active jobs and submits new ones from the queue."""
    while True:
        for job_folder, process in list(active_jobs.items()):
            retcode = process.poll()
            if retcode is not None:  # Job has finished
                del active_jobs[job_folder]
                with open(os.path.join(job_folder, 'FINISHED'), 'w') as f:
                    pass
                if clean_files_flag:
                    clean_out_files(out_folder_path=job_folder, rm_out_files_list=rm_out_files_list)
                if not job_queue.empty():  # Submit new job if queue is not empty
                    next_job = job_queue.get()
                    active_jobs[next_job] = submit_job(next_job, cmd_line)
        time.sleep(1)  # Check job status every second


def split_database(db_path, output_prefix, structures_per_split):
    """Splits an ase.db database into smaller databases."""
    db = connect(db_path)
    total_ids = list(db.get_ids())
    n_splits = math.ceil(len(total_ids) / structures_per_split)  # Calculate number of splits needed
    split_db_paths = []

    for i in range(n_splits):
        output_db_path = f"{output_prefix}_{i}.db"
        split_db_paths.append(output_db_path)
        split_db = connect(output_db_path)
        start_index = i * structures_per_split
        end_index = min((i + 1) * structures_per_split, len(total_ids))
        for j in range(start_index, end_index):
            atoms = db.get_atoms(id=total_ids[j])
            split_db.write(atoms)
        print(f"Created split database: {output_db_path}, containing {end_index - start_index} structures")
    return split_db_paths


def modify_input_file(parameters):
    """Modifies the 'INPUT' file based on the provided parameters."""
    filepath = 'INPUT'
    keyword_found_in_file = {}  # Track if keyword is found in the file
    for keyword in parameters:
        keyword_found_in_file[keyword] = False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        line_modified = False
        for keyword, new_value in parameters.items():
            if line.strip().startswith(keyword):
                modified_lines.append(f"{keyword} {new_value}\n")  # Replace the entire line
                keyword_found_in_file[keyword] = True
                line_modified = True
                break  # Move to the next line after modifying
        if not line_modified:
            modified_lines.append(line)  # Keep original line if not modified

    # Add parameters that were not found in the file
    for keyword, new_value in parameters.items():
        if not keyword_found_in_file[keyword]:
            modified_lines.append(f"{keyword} {new_value}\n")

    with open(filepath, 'w') as f:
        f.writelines(modified_lines)


def prepare_job_directories(db_src_path, common_folder_path, cooking_path, pp_orb_info, has_kpoints=False,
                            csr_key='predicted_data', prep_with_abacus_test_cmd=None):
    cwd_ = os.getcwd()
    os.makedirs(cooking_path, exist_ok=True)
    db_src_path = os.path.abspath(db_src_path)
    with connect(db_src_path) as db_src:
        for a_row in db_src.select():
            an_atoms = a_row.toatoms()
            job_dir = os.path.join(cooking_path, a_row.hpc_id)
            os.makedirs(job_dir, exist_ok=True)
            os.chdir(job_dir)

            # 1. Write Structure
            write_abacus(os.path.join(job_dir, 'STRU'), an_atoms, scaled=False,
                         pp=pp_orb_info['pp'], basis=pp_orb_info['basis'])

            # 2. Copy Common Files
            for item in os.listdir(common_folder_path):
                if os.path.exists(item):
                    continue
                src_file = os.path.join(common_folder_path, item)
                if os.path.isdir(src_file):
                    shutil.copytree(src_file, item)
                else:
                    shutil.copy2(src_file, item)  # copy2 to preserve metadata

            # 3. Write CSR File (If exists in DB)
            # 最小侵入式修改：直接从 data 中读取并写入文件
            if 'csr_info' in a_row.data and csr_key in a_row.data:
                csr_info = a_row.data['csr_info']
                blocks = a_row.data[csr_key]
                matrix_symbol = csr_info.get('matrix_symbol', 'H')

                # 获取文件名并拼接当前 job_dir 路径
                csr_filename = get_abacus_csr_name(matrix_symbol)
                output_csr_path = os.path.join(job_dir, csr_filename)

                write_blocks_to_abacus_csr(
                    matrix_symbol=matrix_symbol,
                    atomic_numbers=an_atoms.get_atomic_numbers(),
                    basis_dict=csr_info['basis_dict'],
                    blocks_dict=blocks,
                    output_path=output_csr_path
                )

            # 4. Modify Input
            modify_input_file({})

            # 5. Run abacus test (if needed)
            if prep_with_abacus_test_cmd not in (None, 'None'):
                prep_with_abacus_test_cmd = prep_with_abacus_test_cmd + f' -j ./'
                os.system(prep_with_abacus_test_cmd)

            # 6. Handle Kpoints (if needed)
            if has_kpoints:
                save_direct_kpoints(kpoints=a_row.data['kpoints'])
                np.save('old_bands.npy', np.array(a_row.data['bands']))
                os.remove('KPT.nscf')
                shutil.copy(src='old_kpoints', dst='KPT.nscf')

            os.chdir(cwd_)
    return


def create_local_handler_file(n_parallel_jobs,
                              pp_orb_info,
                              cmd_line,
                              prep_with_abacus_test_cmd=None,
                              common_folder_path='public',
                              csr_key='predicted_data',
                              local_db_name='sub_structures.db',
                              clean_files_flag=False,
                              rm_out_files_list=[],
                              has_kpoints=False,
                              output_file='local_handler.py'):
    """Create a local_handler.py file based on the provided parameters."""
    file_content = f"""import os
import sys
from ase.calculators.abacus import Abacus

import logging
from dprep.dpdispatcher_tools import run_jobs_locally

# Configure logging for local handler example to 'local_handler_example.log'
logging.basicConfig(filename='local_handler_example.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

run_jobs_locally(
    n_parallel_jobs={n_parallel_jobs},
    local_db_name='{local_db_name}',
    common_folder_path='{common_folder_path}',
    cmd_line='{cmd_line}',
    csr_key='{csr_key}',
    clean_files_flag={clean_files_flag},
    pp_orb_info={pp_orb_info},
    prep_with_abacus_test_cmd="{prep_with_abacus_test_cmd}",
    rm_out_files_list={rm_out_files_list},
    has_kpoints={has_kpoints}
)
"""
    with open(output_file, 'w') as f:
        f.write(file_content)


def run_jobs_locally(n_parallel_jobs, cmd_line,
                     common_folder_path,
                     pp_orb_info,
                     prep_with_abacus_test_cmd=None,
                     csr_key='predicted_data',
                     clean_files_flag=False,
                     local_db_name='sub_structures.db',
                     rm_out_files_list: list = [],
                     has_kpoints=False, ):
    cwd_ = os.getcwd()
    cooking_path = os.path.abspath('cooking')
    common_folder_path = os.path.abspath(common_folder_path)
    prepare_job_directories(db_src_path=local_db_name, common_folder_path=common_folder_path, cooking_path=cooking_path,
                            pp_orb_info=pp_orb_info, has_kpoints=has_kpoints, csr_key=csr_key,
                            prep_with_abacus_test_cmd=prep_with_abacus_test_cmd)

    job_queue = Queue()
    job_folder_path_list = []
    for a_job_folder in os.listdir(cooking_path):
        abs_job_folder = os.path.join(cooking_path, a_job_folder)
        job_queue.put(abs_job_folder)
        job_folder_path_list.append(abs_job_folder)
    n_total_jobs = len(job_folder_path_list)

    # Dictionary to keep track of active jobs
    active_jobs = {}
    # Submit initial jobs
    for _ in range(min(n_parallel_jobs, job_queue.qsize())):
        job_folder = job_queue.get()
        active_jobs[job_folder] = submit_job(job_folder, cmd_line)

    # Start monitoring thread
    monitor_thread = Thread(target=monitor_jobs,
                            args=(job_queue, active_jobs, cmd_line, clean_files_flag, rm_out_files_list), daemon=True)
    monitor_thread.start()

    os.chdir(cwd_)
    start_time = datetime.now()
    while not job_queue.empty() or active_jobs:
        time.sleep(10)
        n_completed_jobs = 0
        for root, dirs, files in os.walk(cooking_path):
            n_completed_jobs += files.count('FINISHED')
        log_remaining_time(start_time=start_time, n_completed_jobs=n_completed_jobs, n_total_jobs=n_total_jobs)


def run_jobs_remotely(n_parallel_machines, resrc_info, machine_info, local_job_para, db_src_path, common_folder_path,
                      pp_orb_info_path, check_file_name='dmrs1_nao.csr', id_name=None):
    cwd_ = os.getcwd()
    db_src_path = os.path.abspath(db_src_path)
    common_folder_path = os.path.abspath(common_folder_path)

    # prepare
    cooking_path = os.path.abspath('cooking')
    if os.path.exists(cooking_path):
        print('Found previous workbase. It will be cleared.')
        shutil.rmtree(cooking_path)
    os.makedirs(cooking_path)
    task_list = []

    with connect(db_src_path) as db:
        total_n_mols = db.count()

    random_idx_list = list(range(total_n_mols))
    random.shuffle(random_idx_list)
    if total_n_mols % n_parallel_machines == 0:
        sub_n_mols = total_n_mols // n_parallel_machines
        actual_machine_used = n_parallel_machines
    else:
        sub_n_mols = math.ceil(total_n_mols / n_parallel_machines)
        actual_machine_used = total_n_mols // sub_n_mols + 1

    local_job_para['pp_orb_info'] = generate_pp_orb_dict(pp_orb_info_path)
    create_local_handler_file(**local_job_para)

    with connect(db_src_path) as src_db:
        if id_name is not None:
            for a_row in src_db.select():
                try:
                    an_id = a_row[id_name]
                    direct_id_name_flag = True
                except:
                    direct_id_name_flag = False
        for i in range(actual_machine_used):
            os.chdir(cooking_path)
            os.makedirs(f'{str(i)}')
            os.chdir(f'{str(i)}')
            shutil.copytree(src=common_folder_path, dst='public')
            shutil.copy(src=os.path.join(cwd_, 'local_handler.py'), dst='local_handler.py')
            with connect('sub_structures.db') as sub_db:
                start_index = i * sub_n_mols
                end_index = (i + 1) * sub_n_mols if i < actual_machine_used - 1 else len(random_idx_list)
                indices_for_sub = random_idx_list[
                                  start_index:end_index]  # Slice the random index list for the current sub
                for real_idx in indices_for_sub:
                    for a_row in src_db.select(id=real_idx + 1):
                        pass
                    an_atoms = a_row.toatoms()  # Convert the data row to 'atoms' object
                    if id_name:
                        if direct_id_name_flag:
                            an_id = a_row[id_name]
                        else:
                            an_id = a_row.data[id_name]
                        hpc_id = an_id
                    else:
                        hpc_id = f'db_seq_id_{real_idx}'
                    sub_db.write(an_atoms, hpc_id=hpc_id, data=a_row.data)
            # task
            a_task = Task(
                command=fr'python local_handler.py 2>&1 ',
                task_work_path=f'{str(i)}/',
                forward_files=[f'{cooking_path}/{str(i)}/*'],
                backward_files=[f'cooking']
            )
            task_list.append(a_task)
    os.chdir(cwd_)
    # submission
    submission = Submission(
        work_base=cooking_path,
        machine=Machine.load_from_dict(machine_dict=machine_info),
        resources=Resources.load_from_dict(resources_dict=resrc_info),
        task_list=task_list,
    )
    submission.run_submission()
    copy_failed_folders(
        source_dir=cooking_path,
        target_folder_name='OUT.ABACUS',
        check_file_name=check_file_name,
        dump_dir=os.path.join(cwd_, 'failed_jobs')
    )