import os.path
import sys
import os

local_root = os.getcwd()
sys.path.append(local_root)

from dptb_dpdispatcher_tools import run_jobs_remotely

# -----------------------------------------------------------------------------------------------------------------
n_parallel_machines = 2
# -----------------------------------------------------------------------------------------------------------------
mach_para = {
    'batch_type': "Shell",
    'context_type': "LocalContext",
    'remote_root': '/root/test_dpdispatcher',
    'remote_profile': {},
    'local_root': os.path.join(local_root, 'cooking')
}

resrc_para = {
    'number_node': 1,
    'cpu_per_node': 64,
    'gpu_per_node': 0,
    'group_size': 1,
    'queue_name': "amd_256",
    'envs': {
        "OMP_STACKSIZE": "4G",
        "OMP_NUM_THREADS": "1",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "MKL_NUM_THREADS": "3"
    },
}

run_jobs_remotely(n_parallel_machines=n_parallel_machines, resrc_info=resrc_para, machine_info=mach_para,
                  db_src_path=r'filtered_2DMat_sample.db',
                  common_folder_path=os.path.join(local_root, 'public'),
                  handler_file_name=r'local_handler.py',
                  sub_db_name=r'sub_structures.db')
