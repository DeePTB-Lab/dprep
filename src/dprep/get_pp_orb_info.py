import os
import json


def generate_pp_orb_dict(directory_path):
    """
    Generates a dictionary of pseudopotential and orbital basis files for elements.

    Args:
        directory_path (str): Path to the directory containing the files

    Returns:
        dict: Dictionary with 'pp' and 'basis' sections for each element
    """
    # Initialize the dictionary structure
    pp_orb_info = {
        'pp': {},
        'basis': {}
    }

    # Get all files in the directory
    try:
        files = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"Error: Directory {directory_path} not found.")
        return None
    except PermissionError:
        print(f"Error: Permission denied for directory {directory_path}.")
        return None

    # Process each file
    for filename in files:
        # Skip directories and non-relevant files
        if os.path.isdir(os.path.join(directory_path, filename)) or not (
                filename.endswith('.upf') or filename.endswith('.orb')  or filename.endswith('.UPF')):
            continue

        # Extract element symbol (first part of filename before underscore)
        element = filename.split('_')[0].split('.')[0].split('-')[0]

        # Categorize files by type
        if filename.endswith('.upf'):
            pp_orb_info['pp'][element] = filename
        elif filename.endswith('.UPF'):
            pp_orb_info['pp'][element] = filename
        elif filename.endswith('.orb'):
            pp_orb_info['basis'][element] = filename

    return pp_orb_info

# excute this file to update the pp orb info
if __name__ == "__main__":
    directory_path = r"E:\deeptb\basis_set_test\new_basis_set"

    # Generate the dictionary
    result = generate_pp_orb_dict(directory_path)

    if result:
        # Print formatted output
        print("default_pp_orb_info = {")
        print("    'pp': {")
        for element, filename in sorted(result['pp'].items()):
            print(f"        '{element}': '{filename}',")
        print("    },")
        print("    'basis': {")
        for element, filename in sorted(result['basis'].items()):
            print(f"        '{element}': '{filename}',")
        print("    }")
        print("}")

        # Optionally save to a Python file
        with open('pp_orb_info.py', 'w') as f:
            f.write("default_pp_orb_info = ")
            f.write(json.dumps(result, indent=4))
            f.write("\n")

        print("\nDictionary also saved to pp_orb_info.py")