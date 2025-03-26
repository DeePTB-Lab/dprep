import os
import sys
from ase.calculators.abacus import Abacus

import logging
from dptb_dpdispatcher_tools import run_jobs_locally

# Configure logging for local handler example to 'local_handler_example.log'
logging.basicConfig(filename='local_handler_example.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


run_jobs_locally(
    n_parallel_jobs=1,
    db_src_path='sub_structures.db',
    common_folder_path='public',
    cmd_line='OMP_NUM_THREADS=1 mpirun -n 24 abacus',
    clean_files_flag=True,
    rm_out_files_list=[]
)

