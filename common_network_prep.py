import os
import re
import tarfile
import zipfile
from torchvision import datasets
from torchvision.models import list_models
import torchvision.models as models

# Initialize Project_Info as a global array
Project_Info = []

def sanitize_string(s):
    """Convert string to lowercase, strip special characters, and replace spaces with underscores."""
    s = s.lower()
    s = re.sub(r'[^a-z0-9_]', '', s)
    return s.replace(' ', '_')

def ensure_directory_exists(dir_path):
    """Ensure the directory exists; create it if it does not and set permissions to 777."""
    os.makedirs(dir_path, exist_ok=True)
    os.chmod(dir_path, 0o777)
    return os.path.abspath(dir_path)

def check_file_exists(file_path):
    """Check if a file exists."""
    return os.path.isfile(file_path)

def is_archive_file(file_path):
    """Check if a file is a valid archive (e.g., zip, tar) that we can extract."""
    return zipfile.is_zipfile(file_path) or tarfile.is_tarfile(file_path)

def setup_project(prj_name, project_folder, model_type, model_py_file=None, model_pth_file=None, torch_vision_model=None, dataset_type=None, custom_dataset=None, torch_vision_dataset=None):
    # Ensure Project Name and Project Folder are provided
    if not prj_name or not project_folder:
        raise ValueError("Project Name and Project Folder are required")

    # Sanitize project name and set project info
    prj_name_stripped = sanitize_string(prj_name)
    display_name = prj_name
    Project_Info.append({"Display_Name": display_name, "Stripped_Name": prj_name_stripped})

    # Ensure the directory exists
    folder_path = ensure_directory_exists(project_folder)
    Project_Info.append({"Folder": folder_path})

    # Validate Model Type
    valid_model_types = ["untrained", "custom_pretrained", "torch_vision_pretrained"]
    if model_type not in valid_model_types:
        raise ValueError(f"Model Type must be one of {valid_model_types}")

    # Handle Model Type specific requirements
    if model_type in ["untrained", "custom_pretrained"]:
        # Ensure Model Py File is provided and exists
        if not model_py_file or not check_file_exists(os.path.join(project_folder, model_py_file)):
            raise FileNotFoundError(f"Model Py File '{model_py_file}' does not exist in '{project_folder}'")
        Project_Info.append({"Model_Py_File": model_py_file})

    if model_type == "custom_pretrained":
        # Ensure Model Pth File is provided and exists
        if not model_pth_file or not check_file_exists(os.path.join(project_folder, model_pth_file)):
            raise FileNotFoundError(f"Model Pth File '{model_pth_file}' does not exist in '{project_folder}'")
        Project_Info.append({"Model_Pth_File": model_pth_file})

    if model_type == "torch_vision_pretrained":
        # Ensure Torch Vision Model is provided and valid
        available_models = list_models(module=models)
        if not torch_vision_model or torch_vision_model not in available_models:
            raise ValueError(f"Torch Vision Model must be one of: {available_models}")
        Project_Info.append({"Torch_Vision_Model": torch_vision_model})

    # Handle Dataset requirements for Untrained models
    if model_type == "untrained":
        if dataset_type not in ["torch_vision_dataset", "custom_dataset"]:
            raise ValueError("Dataset Type must be either 'Torch Vision' or 'Custom Dataset' for Untrained models")
        Project_Info.append({"Dataset_Type": dataset_type})

        if dataset_type == "custom_dataset":
            # Ensure Custom Dataset file exists and is an archive
            custom_dataset_path = os.path.join(project_folder, custom_dataset)
            if not custom_dataset or not check_file_exists(custom_dataset_path) or not is_archive_file(custom_dataset_path):
                raise FileNotFoundError(f"Custom Dataset '{custom_dataset}' must exist in '{project_folder}' and be an archive file (zip or tar)")
            Project_Info.append({"Custom_Dataset": custom_dataset})

        elif dataset_type == "torch_vision_dataset":
            # Ensure Torch Vision Dataset is valid
            available_datasets = [cls_name.lower() for cls_name in dir(datasets) if not cls_name.startswith('_')]
            if not torch_vision_dataset or torch_vision_dataset.lower() not in available_datasets:
                raise ValueError(f"Torch Vision Dataset must be one of: {available_datasets}")
            Project_Info.append({"Torch_Vision_Dataset": torch_vision_dataset})

    # Output the Project Info array
    return Project_Info
