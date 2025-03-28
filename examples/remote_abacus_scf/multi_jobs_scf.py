import os

from dprep.dpdispatcher_tools import run_jobs_remotely

# -----------------------------------------------------------------------------------------------------------------
n_parallel_machines = 2
local_job_para = {
    'n_parallel_jobs': 2,
    'cmd_line': 'OMP_NUM_THREADS=1 mpirun -n 16 abacus',
    'clean_files_flag': True,
    'rm_out_files_list': ['PP_ORB', 'ABACUS-CHARGE-DENSITY.restart', 'SPIN1_CHG.cube']
}
# -----------------------------------------------------------------------------------------------------------------
local_root = os.getcwd()
mach_para = {
    "batch_type": "Bohrium",
    "context_type": "Bohrium",
    'remote_root': '/root/test_dpdispatcher',
    'remote_profile': {
        "email": 'xx',
        "password": 'xx',
        "project_id": 0,
        "input_data": {
            "job_type": "container",
            "log_file": "log",
            "job_name": "test_dprep",
            "disk_size": 200,
            "scass_type": 'c64_m128_cpu',
            "platform": 'ali',
            "image_name": "registry.dp.tech/dptech/dp/native/prod-11729/dprep:0328v0"
        }
    },
    'local_root': r'./'
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
                  db_src_path=r'test_4_structures.db',
                  common_folder_path=os.path.join(local_root, 'public'),
                  pp_orb_info_path=os.path.join(local_root, 'public', 'PP_ORB'),
                  local_job_para=local_job_para)
