from finn.util.basic import pynq_part_map
from torchvision import datasets
import os

# Output directories
WORKING_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)),"outputs/")

available_boards = list(pynq_part_map.keys())
sample_pretrained_models = ["alexnet_mnist", "resnet_mnist", "alexnet_cifar10", "resnet_cifar10"]
sample_untrained_models = ["alexnet", "resnet"]
finn_pretrained_models = ["cnv_1w1a", "cnv_1w2a", "cnv_2w2a", "lfc_1w1a", "lfc_1w2a", "sfc_1w1a", "sfc_1w2a", "sfc_2w2a", "tfc_1w1a", "tfc_1w2a", "tfc_2w2a", "quant_mobilenet_v1_4b"]
available_datasets = [cls_name.lower() for cls_name in dir(datasets) if not cls_name.startswith('_')]
valid_model_types = ["untrained", "sample_untrained", "custom_pretrained", "sample_pretrained", "finn_pretrained"]

def validate_file(file, allowed_extensions):
    """
    Validate the uploaded file for presence and allowed extensions.
    """
    if not file:
        return False, "No file provided."
    filename = file.filename
    if "." not in filename or filename.rsplit(".", 1)[1].lower() not in allowed_extensions:
        return False, f"Invalid file type. Allowed extensions are: {', '.join(allowed_extensions)}."
    return True, "File is valid."


import shutil
import os
import re
from datetime import datetime
import tarfile
import zipfile

# List to store logs
log_list = []


def log_message(message, level="info"):
    """
    Log a message, with support for info, warning, and error levels.
    If the level is "error", this function raises a ValueError with the message.

    Parameters:
        message (str): The message to log.
        level (str): The level of the message ("info", "warning", or "error").
    """
    global log_list
    log_entry = f"[{level.upper()}] {message}"
    log_html_entry = f"<b>[{level.upper()}]</b> {message}<br>"

    # Add log entry to the list
    log_list.append(log_html_entry)

    # Print log entry for immediate feedback
    print(log_entry)

    # Raise an error if the level is "error"
    if level == "error":
        raise ValueError(message)


def get_logs():
    """
    Get all the logs concatenated into a single string and clear the log list.

    Returns:
        str: All log messages concatenated together, separated by new lines.
    """
    global log_list
    # Concatenate all logs into a single string
    logs = "\n".join(log_list)

    # Clear the log list
    log_list.clear()

    return logs


def sanitize_string(s):
    """
    Convert a string to lowercase, strip special characters, and replace spaces with underscores.

    Parameters:
        s (str): The input string to sanitize.

    Returns:
        str: The sanitized string.
    """
    s = s.lower()
    s = s.replace(' ', '_')
    s = re.sub(r'[^a-z0-9_]', '', s)
    return s


def ensure_directory_exists(dir_path):
    """
    Ensure the specified directory exists; create it if it does not, including required subdirectories.
    Set permissions to 777 for the main directory and its contents recursively.

    Parameters:
        dir_path (str): The path of the directory to ensure exists.

    Returns:
        str: The absolute path of the directory.
    """
    os.makedirs(dir_path, exist_ok=True)

    # List of required subdirectories
    subdirs = ['dataset', 'src', 'checkpoints', 'output']
    dirs_to_clear = ['checkpoints', 'output']

    # Create each subdirectory if it doesn’t exist
    for subdir in subdirs:
        subdir_path = os.path.join(dir_path, subdir)
        os.makedirs(subdir_path, exist_ok=True)

    # Set permissions recursively for main directory and all subdirectories/files
    for root, dirs, files in os.walk(dir_path):
        os.chmod(root, 0o777)
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            if 'zip' not in f:
                os.chmod(os.path.join(root, f), 0o777)

    # Clear all contents in the specified directories
    for clear_dir in dirs_to_clear:
        clear_path = os.path.join(dir_path, clear_dir)
        for filename in os.listdir(clear_path):
            file_path = os.path.join(clear_path, filename)
            try:
                if (os.path.isfile(file_path) or os.path.islink(file_path)) and 'zip' not in file_path:
                    os.unlink(file_path)  # Remove file or link
                elif os.path.isdir(file_path) and 'output' not in file_path:
                    shutil.rmtree(file_path)  # Remove directory and all contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    return os.path.abspath(dir_path)


def check_file_exists(file_path):
    """
    Check if a file exists at the specified path.

    Parameters:
        file_path (str): The path of the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)


def is_archive_file(file_path):
    """
    Check if a file is a valid archive (zip or tar).

    Parameters:
        file_path (str): The path of the file to check.

    Returns:
        bool: True if the file is a valid archive, False otherwise.
    """
    return zipfile.is_zipfile(file_path) or tarfile.is_tarfile(file_path)

def set_onnx_checkpoint(project_info, suffix):
    """
    Generates the export path for ONNX files based on a specified suffix.

    Parameters:
        project_info (dict): Dictionary containing project information (e.g., 'Folder' and 'Stripped_Name').
        suffix (str): The suffix to append to the exported ONNX filename (e.g., "model1" for "model1_export.onnx").

    Returns:
        str: The full path to the export file.
    """
    log_message(f"Saving Checkpoint: {suffix}")
    suffix = sanitize_string(suffix)
    filename = f"{project_info['Stripped_Name']}_{suffix}.onnx"
    return os.path.join(project_info['Folder'], "checkpoints", filename)
