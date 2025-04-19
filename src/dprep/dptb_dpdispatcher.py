import json
import os
import re
import shutil

from dpdispatcher import Task, Submission, Machine, Resources
import copy
import itertools


def merge_parameters(private_para, public_para):
    """
    Merges private and public parameters by generating all combinations
    of list values in private_para and merging them into public_para.

    :param private_para: Dictionary containing private parameters with possible list values.
    :param public_para: Dictionary containing public parameters.
    :return: List of dictionaries with all combinations merged.
    """

    # Helper function to recursively find all paths leading to list values
    def find_list_paths(d, current_path=[]):
        paths = []
        for k, v in d.items():
            new_path = current_path + [k]
            if isinstance(v, dict):
                paths.extend(find_list_paths(v, new_path))
            elif isinstance(v, list):
                paths.append((new_path, v))
        return paths

    # Merge other parts of private_para that don't contain lists
    def merge_remaining(private_sub, merged_sub):
        for k, v in private_sub.items():
            if isinstance(v, dict):
                if k not in merged_sub or not isinstance(merged_sub[k], dict):
                    merged_sub[k] = {}
                merge_remaining(v, merged_sub[k])
            elif not isinstance(v, list):  # Skip lists as they are already processed
                merged_sub[k] = v

    # Find all paths with list values in private_para
    list_paths = find_list_paths(private_para)

    # Extract the lists and their corresponding paths
    keys = [path for path, values in list_paths]
    values_lists = [values for path, values in list_paths]

    # Generate all possible combinations
    combinations = list(itertools.product(*values_lists))

    for combo in combinations:
        # Start with a deep copy of public_para
        merged = copy.deepcopy(public_para)
        # Insert each value from the combination into the correct path in merged dict
        for path, value in zip(keys, combo):
            current_level = merged
            for key in path[:-1]:
                if key not in current_level or not isinstance(current_level[key], dict):
                    current_level[key] = {}
                current_level = current_level[key]
            current_level[path[-1]] = value
        merge_remaining(private_para, merged)
        a_private_name = '_'.join([str(x) for x in combo])
        yield merged, a_private_name


# patience * nsamples / batch_size > 2000
# Here nsamples=2500
def maintain_patience(merged_para_dict):
    patience = merged_para_dict["train_options"]["batch_size"]
    merged_para_dict["train_options"]["lr_scheduler"].update({"patience": patience})
    return merged_para_dict



def parse_orbital_files(folder_path):
    """
    Parse orbital filenames to extract element, cutoff, and basis information

    Args:
        folder_path (str): Path to the directory containing orbital files

    Returns:
        tuple: (cutoff_dict, basis_dict) containing the extracted information
    """
    cutoff_dict = {}
    basis_dict = {}

    # Get all files in the directory
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
        return {}, {}
    except PermissionError:
        print(f"Error: No permission to access folder '{folder_path}'.")
        return {}, {}

    # More flexible pattern that makes "gga" optional but requires underscore after element
    pattern = r"([A-Za-z]+)_(?:gga_)?(\d+)au_(\d+\.\d+|\d+)Ry_(\w+)\.orb"

    for file in files:
        match = re.match(pattern, file)
        if match:
            element = match.group(1)
            au_value = match.group(2)  # Uncomment if you need this value
            # cutoff = float(match.group(3))
            basis = match.group(4)

            # Store the cutoff value
            cutoff_dict[element] = int(au_value)

            # Store the basis value
            basis_dict[element] = basis

    return cutoff_dict, basis_dict


class DPTBDpdispatcher:
    def __init__(self,
                 private_para_dict: dict,
                 public_para_dict: dict,
                 machine_info: dict,
                 resrc_info: dict,
                 cmd_line: str=fr'dptb train input.json -o ./output 2>&1 ',
                 old_ckpt_path: str=None):
        self.private_para_dict = private_para_dict
        self.public_para_dict = public_para_dict
        self.machine_info = machine_info
        self.resrc_info = resrc_info
        self.workbase = os.getcwd()
        self.cmd_line = cmd_line
        if old_ckpt_path:
            self.old_ckpt_name = os.path.basename(old_ckpt_path)
            self.old_ckpt_path = os.path.abspath(old_ckpt_path)
        else:
            self.old_ckpt_path = None

    def find_largest_event_file(self, directory='.'):
        largest_file = None
        largest_size = -1
        for file in os.listdir(directory):
            if file.startswith('events.out.tfevents'):
                file_path = os.path.join(directory, file)
                file_size = os.path.getsize(file_path)
                if file_size > largest_size:
                    largest_file = file
                    largest_size = file_size
        return largest_file

    def prepare_workbase(self):
        self.task_list = []
        self.path_raw = os.path.abspath('raw')
        self.path_raw_job_paths = []
        os.makedirs(exist_ok=True, name=self.path_raw)

        for a_merged_para, job_name in merge_parameters(private_para=self.private_para_dict, public_para=self.public_para_dict):
            os.chdir(self.path_raw)
            os.makedirs(job_name, exist_ok=True)
            os.chdir(job_name)
            self.path_raw_job_paths.append(os.getcwd())
            # a_merged_para = maintain_patience(a_merged_para)
            with open(r"input.json", 'w') as f:
                json.dump(a_merged_para, f, indent=4)
            if self.old_ckpt_path:
                shutil.copy(src=self.old_ckpt_path, dst=self.old_ckpt_name)
            a_task = Task(command=self.cmd_line,
                          task_work_path=f'{job_name}/',
                          forward_files=[f'{self.path_raw}/{job_name}/*'],
                          backward_files=['tensorboard_logs/*', 'output/*'])
            self.task_list.append(a_task)

        os.chdir(self.workbase)

    def run_a_batch(self):
        machine = Machine.load_from_dict(machine_dict=self.machine_info)
        resources = Resources.load_from_dict(resources_dict=self.resrc_info)
        submission = Submission(work_base=f'{self.path_raw}',
                                machine=machine,
                                resources=resources,
                                task_list=self.task_list,
                                forward_common_files=[],
                                backward_common_files=[]
                                )
        submission.run_submission(check_interval=60, clean=True)

    def post_process(self):
        self.path_cooked = os.path.abspath('cooked')
        if os.path.exists(self.path_cooked):
            shutil.rmtree(self.path_cooked)
        os.makedirs(name=self.path_cooked)
        os.chdir(self.path_cooked)
        os.makedirs('ckpt')
        self.ckpt_path = os.path.abspath('ckpt')
        os.makedirs('events')
        self.events_path = os.path.abspath('events')
        for a_job_path in self.path_raw_job_paths:
            a_job_name = os.path.split(a_job_path)[-1]
            os.chdir(a_job_path)
            os.chdir('tensorboard_logs')
            largest_file = self.find_largest_event_file()
            new_event_folder_path = os.path.join(self.events_path, a_job_name)
            os.makedirs(new_event_folder_path)
            shutil.copy(src=largest_file, dst=os.path.join(new_event_folder_path, largest_file))
            shutil.copy(src=os.path.join(a_job_path, 'output', 'checkpoint', 'nnenv.best.pth'),
                        dst=os.path.join(self.ckpt_path, f'{a_job_name}.pth'))
        os.chdir(self.workbase)

    def run_with_dpdispatcher(self):
        self.prepare_workbase()
        self.run_a_batch()
        self.post_process()
