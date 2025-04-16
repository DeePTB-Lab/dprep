import os

from dprep.dpdispatcher_tools import run_jobs_remotely

# -----------------------------------------------------------------------------------------------------------------
n_parallel_machines = 2
local_job_para = {
    'n_parallel_jobs': 2,
    'cmd_line': 'OMP_NUM_THREADS=1 mpirun -n 12 abacus',
}
# -----------------------------------------------------------------------------------------------------------------
local_root = os.getcwd()
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
        "MKL_NUM_THREADS": "1"
    },
}


run_jobs_remotely(n_parallel_machines=n_parallel_machines, resrc_info=resrc_para, machine_info=mach_para,
                  db_src_path=r'test_4_structures.db',
                  common_folder_path=os.path.join(local_root, 'public'),
                  pp_orb_info_path=os.path.join(local_root, 'public', 'PP_ORB'),
                  local_job_para=local_job_para)
