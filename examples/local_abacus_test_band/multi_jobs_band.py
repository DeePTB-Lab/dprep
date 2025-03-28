import os

from dprep.dpdispatcher_tools import run_jobs_remotely

# -----------------------------------------------------------------------------------------------------------------
n_parallel_machines = 2
local_job_para = {
    'n_parallel_jobs': 2,
    'prep_with_abacus_test_cmd': "abacustest model band prepare -c 'OMP_NUM_THREADS=1 mpirun -np 12 abacus | tee out.log'",
    'cmd_line': 'bash run.sh >>run_info.log ', # special line, be careful
    'clean_files_flag': True,
    'rm_out_files_list': ['PP_ORB', 'ABACUS-CHARGE-DENSITY.restart', 'SPIN1_CHG.cube']
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
}

run_jobs_remotely(n_parallel_machines=n_parallel_machines, resrc_info=resrc_para, machine_info=mach_para,
                  db_src_path=r'test_4_structures.db',
                  common_folder_path=os.path.join(local_root, 'public'),
                  pp_orb_info_path=os.path.join(local_root, 'public', 'PP_ORB'),
                  local_job_para=local_job_para)
