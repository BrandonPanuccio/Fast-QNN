from brevitas.export import export_qonnx
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import torch

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.merge_onnx_models import MergeONNXModels

from helper import *
from finn_flow import *


from helper import *

progress_step=""
Project_Info = []
input_model = None
input_model_shape = None
export_onnx_path = ""
model = ""

def setup_project(prj_name, brd_name, model_type, project_folder=None, model_py_file=None, model_pth_file=None,
                  sample_untrained_model=None, sample_pretrained_model=None, finn_pretrained_model=None,
                  dataset_type=None, custom_dataset=None, torch_vision_dataset=None):
    """
    Set up a project with a specified structure, including creating necessary directories,
    validating model type, and checking for necessary files.

    Parameters:
        prj_name (str): The name of the project.
        brd_name (str): The name of the target board for the project (e.g., PYNQ board).
        model_type (str): The type of model ('untrained', 'sample_untrained', 'custom_pretrained',
                          'sample_pretrained', or 'finn_pretrained').
        project_folder (str, optional): The main folder for the project. A new folder is generated if not provided.
        model_py_file (str, optional): The filename of the Python script defining the model architecture, required
                                       for 'untrained' and 'custom_pretrained' models.
        model_pth_file (str, optional): The filename of the .pth file with pre-trained weights, required for
                                        'custom_pretrained' models.
        sample_untrained_model (str, optional): The name of the sample untrained model to load, required for
                                                'sample_untrained' models.
        sample_pretrained_model (str, optional): The name of the sample pretrained model to load, required for
                                                 'sample_pretrained' models.
        finn_pretrained_model (str, optional): The name of the FINN pretrained model to load, required for
                                               'finn_pretrained' models.
        dataset_type (str, optional): Type of dataset for 'untrained' models ('torch_vision_dataset' or
                                      'custom_dataset').
        custom_dataset (str, optional): Path to the custom dataset file for 'untrained' models with 'custom_dataset'
                                        dataset type.
        torch_vision_dataset (str, optional): Name of the TorchVision dataset class for 'untrained' models with
                                              'torch_vision_dataset' dataset type.

    Returns:
        dict: A dictionary with project setup information including project name, folder path, model details,
              and dataset information.
    """
    global progress_step
    log_message("Setting up project")
    progress_step = "Project Setup"
    working_folder = "/home/fastqnn/finn/notebooks/Fast-QNN/outputs/txaviour/"
    prj_info = {}

    # Ensure Project Name is provided
    if not prj_name:
        log_message("Project Name is required", level="error")

    # Sanitize project name and set project info
    prj_name_stripped = sanitize_string(prj_name)
    display_name = prj_name
    prj_info["Display_Name"] = display_name
    prj_info["Stripped_Name"] = prj_name_stripped

    if not project_folder:
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # for production mode
        timestamp = "0"
        project_folder = f"{working_folder}{prj_name_stripped}_{timestamp}"

    folder_path = ensure_directory_exists(project_folder)
    prj_info["Folder"] = folder_path

    if brd_name in available_boards:
        prj_info["Board_name"] = brd_name
    else:
        log_message(f"'{brd_name}' is not a valid board name. Available board names are: {available_boards}",
                    level="error")

    # Validate Model Type
    if model_type not in valid_model_types:
        log_message(f"Model Type must be one of {valid_model_types}", level="error")
    prj_info['Model_Type'] = model_type

    # Handle Model Type specific requirements
    if model_type in ["untrained", "custom_pretrained"]:
        if not model_py_file or not check_file_exists(os.path.join(project_folder, 'src', model_py_file)):
            log_message(f"Model Py File '{model_py_file}' does not exist in '{project_folder}'", level="error")
        prj_info["Model_Py_File"] = model_py_file

    if model_type == "custom_pretrained":
        if not model_pth_file or not check_file_exists(os.path.join(project_folder, 'src', model_pth_file)):
            log_message(f"Model Pth File '{model_pth_file}' does not exist in '{project_folder}'", level="error")
        prj_info["Model_Pth_File"] = model_pth_file

    if model_type == "sample_pretrained":
        available_models = ["alexnet_mnist", "resnet_mnist", "alexnet_cifar10", "resnet_cifar10"]
        if not sample_pretrained_model or sample_pretrained_model not in available_models:
            log_message(f"Sample Pretrained Model must be one of: {available_models}", level="error")
        prj_info["Pretrained_Model"] = sample_pretrained_model

    if model_type == "sample_untrained":
        available_models = ["alexnet", "resnet"]
        if not sample_untrained_model or sample_untrained_model not in available_models:
            log_message(f"Sample Untrained Model must be one of: {available_models}", level="error")
        prj_info["Sample_Untrained_Model"] = sample_untrained_model

    if model_type == "finn_pretrained":
        available_models = ["cnv_1w1a", "cnv_1w2a", "cnv_2w2a", "lfc_1w1a", "lfc_1w2a", "sfc_1w1a", "sfc_1w2a",
                            "sfc_2w2a", "tfc_1w1a", "tfc_1w2a", "tfc_2w2a", "quant_mobilenet_v1_4b"]
        if not finn_pretrained_model or finn_pretrained_model not in available_models:
            log_message(f"Finn Pretrained Model must be one of: {available_models}", level="error")
        prj_info["Pretrained_Model"] = finn_pretrained_model

    # Handle Dataset requirements for Untrained models
    if model_type == "untrained" or model_type == "sample_untrained":
        if dataset_type not in ["torch_vision_dataset", "custom_dataset"]:
            log_message("Dataset Type must be either 'Torch Vision' or 'Custom Dataset' for Untrained models",
                        level="error")
        prj_info["Dataset_Type"] = dataset_type

        if dataset_type == "custom_dataset":
            custom_dataset_path = os.path.join(project_folder, 'dataset', custom_dataset)
            if not custom_dataset or not check_file_exists(custom_dataset_path) or not is_archive_file(
                    custom_dataset_path):
                log_message(
                    f"Custom Dataset '{custom_dataset}' must exist in '{project_folder}' and be an archive file (zip or tar)",
                    level="error")
            prj_info["Custom_Dataset"] = custom_dataset

        elif dataset_type == "torch_vision_dataset":
            available_datasets = [cls_name.lower() for cls_name in dir(datasets) if not cls_name.startswith('_')]
            if not torch_vision_dataset or torch_vision_dataset.lower() not in available_datasets:
                log_message(f"Torch Vision Dataset must be one of: {available_datasets}", level="error")
            prj_info["Torch_Vision_Dataset"] = torch_vision_dataset

    log_message(f"Project setup complete. {prj_name} has been initialized.")
    return prj_info


from finn.util.test import get_test_model_trained


def load_pretrained_model(model_name, model_type, src_folder, initial_channels=3, max_size=4096):
    """
    Loads a pre-trained model from TorchVision and ensures all downloads are in the src folder of the Project.

    Parameters:
        model_type:
        initial_channels:
        max_size:
        model_name (str): The name of the pre-trained model to load (e.g., 'alexnet', 'resnet50').
        src_folder (str): The folder where model downloads will be stored. Default is 'src'.

    Returns:
        torch.nn.Module: The loaded pre-trained model.
    """
    log_message(f"Loading {model_type} Model: {model_name}")
    # Ensure the src folder exists
    os.makedirs(src_folder, exist_ok=True)
    # Set TORCH_HOME to the src folder to store the model downloads there
    os.environ['TORCH_HOME'] = src_folder
    pretrained_model = None
    if model_type == "sample_pretrained":
        # Load sample pretrained model
        pass
    elif model_type == "finn_pretrained":
        if model_name == "cnv_1w1a":
            pretrained_model = get_test_model_trained("CNV", 1, 1)
        elif model_name == "cnv_1w2a":
            pretrained_model = get_test_model_trained("CNV", 1, 2)
        elif model_name == "cnv_2w2a":
            pretrained_model = get_test_model_trained("CNV", 2, 2)
        elif model_name == "lfc_1w1a":
            pretrained_model = get_test_model_trained("LFC", 1, 1)
        elif model_name == "lfc_1w2a":
            pretrained_model = get_test_model_trained("LFC", 1, 2)
        elif model_name == "sfc_1w1a":
            pretrained_model = get_test_model_trained("SFC", 1, 1)
        elif model_name == "sfc_1w2a":
            pretrained_model = get_test_model_trained("SFC", 1, 2)
        elif model_name == "sfc_2w2a":
            pretrained_model = get_test_model_trained("SFC", 2, 2)
        elif model_name == "tfc_1w1a":
            pretrained_model = get_test_model_trained("TFC", 1, 1)
        elif model_name == "tfc_1w2a":
            pretrained_model = get_test_model_trained("TFC", 1, 2)
        elif model_name == "tfc_2w2a":
            pretrained_model = get_test_model_trained("TFC", 2, 2)
        elif model_name == "quant_mobilenet_v1_4b":
            pretrained_model = get_test_model_trained("mobilenet", 4, 4)
    # List of common input shapes to test first (both square and non-square)
    common_shapes = [
        (1, initial_channels, 32, 32),  # Typical for many models
        (1, initial_channels, 28, 28),  # Typical for many models
        (1, initial_channels, 224, 224),  # Typical for many models
        (1, initial_channels, 299, 299),  # For models like Inception
        (1, initial_channels, 128, 128),  # Smaller size
        (1, initial_channels, 256, 256),  # Larger square size
        (1, initial_channels, 320, 240),  # Common non-square size
        (1, initial_channels, 240, 320),  # Non-square (swapped dimensions)
        (1, initial_channels, 256, 128),  # Non-square
        (1, initial_channels, 128, 256),  # Non-square (swapped)
    ]

    # Try common shapes first
    for shape in common_shapes:
        try:
            dummy_input = torch.rand(*shape)
            pretrained_model(dummy_input)
            log_message(f"Compatible common input shape found: {shape}")
            return pretrained_model, shape
        except RuntimeError:
            continue

    # If no common shape worked, test all possible square and non-square shapes up to max_size
    for width in range(1, max_size + 1, 1):  # Step by 16 for efficiency
        for height in range(1, max_size + 1, 1):
            try:
                dummy_input = torch.rand(1, initial_channels, width, height)
                pretrained_model(dummy_input)
                pretrained_model_input_shape = (1, initial_channels, width, height)
                log_message(f"Compatible input shape found: {pretrained_model_input_shape}")
                return pretrained_model, pretrained_model_input_shape
            except RuntimeError:
                continue

    log_message("Could not determine a compatible input shape within the specified range.", level="error")

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


from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (GiveReadableTensorNames,
                                          GiveUniqueNodeNames,
                                          RemoveStaticGraphInputs,
                                          RemoveUnusedTensors, GiveUniqueParameterTensors, SortGraph)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts


def tidy_up_transforms(input_tidy_model, save_name):
    """
    Applies a series of transformations to a model and saves the resulting model.

    Parameters:
        input_tidy_model (ModelWrapper): The model to transform.
        save_name (str): The path to save the transformed model.

    Returns:
        ModelWrapper: The transformed model.
    """

    # Apply transformations
    input_tidy_model = input_tidy_model.transform(GiveUniqueParameterTensors())
    input_tidy_model = input_tidy_model.transform(InferShapes())
    input_tidy_model = input_tidy_model.transform(FoldConstants())
    input_tidy_model = input_tidy_model.transform(GiveUniqueNodeNames())
    input_tidy_model = input_tidy_model.transform(GiveReadableTensorNames())
    input_tidy_model = input_tidy_model.transform(InferDataTypes())
    input_tidy_model = input_tidy_model.transform(RemoveStaticGraphInputs())

    # Save the transformed model
    input_tidy_model.save(save_name)

    return input_tidy_model
app = Flask(__name__)
CORS(app)

# Output directories
WORKING_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)),"outputs/")

@app.route('/setupproject', methods=['POST'])
def api_setup():
    """
    Handle the submission of project setup data and files.
    """
    global Project_Info
    global progress_step
    progress_step = "Project Setup"
    try:
        # Extract form data
        prj_name = request.form.get("prj_name")
        brd_name = request.form.get("brd_name")
        model_type = request.form.get("model_type")
        dataset_type = request.form.get("dataset_type")
        sample_untrained_model = request.form.get("sample_untrained_model", "").strip()
        sample_pretrained_model = request.form.get("sample_pretrained_model", "").strip()
        finn_pretrained_model = request.form.get("finn_pretrained_model", "").strip()
        torch_vision_dataset = request.form.get("torch_vision_dataset", "").strip()

        # Debugging logs
        print(f"[DEBUG] Received prj_name: {prj_name}, brd_name: {brd_name}, model_type: {model_type}, dataset_type: {dataset_type}")
        print(f"[DEBUG] Received sample_untrained_model: {sample_untrained_model}")
        print(f"[DEBUG] Received sample_pretrained_model: {sample_pretrained_model}")
        print(f"[DEBUG] Received finn_pretrained_model: {finn_pretrained_model}")
        print(f"[DEBUG] Received torch_vision_dataset: {torch_vision_dataset}")

        # Validate required fields
        if not prj_name or not brd_name or not model_type:
            return jsonify({"error": "Missing required fields."}), 400

        # Validate Board Name
        if brd_name not in available_boards:
            return jsonify({"error": f"Invalid board name '{brd_name}'. Available board names are: {', '.join(available_boards)}"}), 400

        # Validate Model Type
        valid_model_types = ["untrained", "sample_untrained", "custom_pretrained", "sample_pretrained", "finn_pretrained"]
        if model_type not in valid_model_types:
            return jsonify({"error": f"Invalid model type '{model_type}'. Model type must be one of {valid_model_types}"}), 400

        # Handle Model Type specific requirements and validate corresponding fields
        model_py_file_path = None
        model_pth_file_path = None
        custom_dataset_path = None

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # for production mode
        timestamp = "0"
        prj_name_stripped = sanitize_string(prj_name)
        project_folder = f"{WORKING_FOLDER}{prj_name_stripped}_{timestamp}"
        folder_path = ensure_directory_exists(project_folder)
        model_dir = os.path.join(folder_path, "src")
        dataset_dir = os.path.join(folder_path, "dataset")

        if model_type in ["untrained", "custom_pretrained"]:
            model_py_file = request.files.get("model_py_file")
            if not model_py_file:
                return jsonify({"error": "Model Python file is required for 'untrained' or 'custom_pretrained' model types."}), 400
            valid, message = validate_file(model_py_file, ["py"])
            if not valid:
                return jsonify({"error": message}), 400
            model_py_file_path = os.path.join(model_dir, f"{prj_name}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
            model_py_file.save(model_py_file_path)

        if model_type == "custom_pretrained":
            model_pth_file = request.files.get("model_pth_file")
            if not model_pth_file:
                return jsonify({"error": "Model PTH file is required for 'custom_pretrained' model type."}), 400
            valid, message = validate_file(model_pth_file, ["pth"])
            if not valid:
                return jsonify({"error": message}), 400
            model_pth_file_path = os.path.join(model_dir, f"{prj_name}_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            model_pth_file.save(model_pth_file_path)

        if model_type == "sample_pretrained":
            available_models = ["alexnet_mnist", "resnet_mnist", "alexnet_cifar10", "resnet_cifar10"]
            if not sample_pretrained_model or sample_pretrained_model not in available_models:
                return jsonify({"error": f"Invalid sample pretrained model '{sample_pretrained_model}'. Available models are: {', '.join(available_models)}"}), 400

        if model_type == "sample_untrained":
            available_models = ["alexnet", "resnet"]
            if not sample_untrained_model or sample_untrained_model not in available_models:
                return jsonify({"error": f"Invalid sample untrained model '{sample_untrained_model}'. Available models are: {', '.join(available_models)}"}), 400

        if model_type == "finn_pretrained":
            available_models = ["cnv_1w1a", "cnv_1w2a", "cnv_2w2a", "lfc_1w1a", "lfc_1w2a", "sfc_1w1a", "sfc_1w2a", "sfc_2w2a", "tfc_1w1a", "tfc_1w2a", "tfc_2w2a", "quant_mobilenet_v1_4b"]
            if not finn_pretrained_model or finn_pretrained_model not in available_models:
                return jsonify({"error": f"Invalid FINN pretrained model '{finn_pretrained_model}'. Available models are: {', '.join(available_models)}"}), 400

        # Validate Dataset Type and handle dataset-related requirements
        if model_type in ["untrained", "sample_untrained"]:
            if dataset_type not in ["torch_vision_dataset", "custom_dataset"]:
                return jsonify({"error": "Invalid dataset type. Dataset type must be either 'torch_vision_dataset' or 'custom_dataset'."}), 400

            if dataset_type == "custom_dataset":
                custom_dataset = request.files.get("custom_dataset")
                if not custom_dataset:
                    return jsonify({"error": "Custom dataset file is required when dataset type is 'custom_dataset'."}), 400
                valid, message = validate_file(custom_dataset, ["zip", "tar"])
                if not valid:
                    return jsonify({"error": message}), 400
                custom_dataset_path = os.path.join(dataset_dir, f"{prj_name}_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
                custom_dataset.save(custom_dataset_path)

            elif dataset_type == "torch_vision_dataset":
                if not torch_vision_dataset or torch_vision_dataset.lower() not in available_datasets:
                    return jsonify({"error": f"Invalid Torch Vision dataset '{torch_vision_dataset}'. Available datasets are: {', '.join(available_datasets)}"}), 400
        try:
            # Call setup_project() with the gathered inputs
            Project_Info = setup_project(
                prj_name=prj_name,
                brd_name=brd_name,
                model_type=model_type,
                project_folder=folder_path,  # Let setup_project create the default folder
                model_py_file=model_py_file_path if model_py_file_path else None,
                model_pth_file=model_pth_file_path if model_pth_file_path else None,
                sample_untrained_model=sample_untrained_model if model_type == "sample_untrained" else None,
                sample_pretrained_model=sample_pretrained_model if model_type == "sample_pretrained" else None,
                finn_pretrained_model=finn_pretrained_model if model_type == "finn_pretrained" else None,
                dataset_type=dataset_type,
                custom_dataset=custom_dataset_path if dataset_type == "custom_dataset" else None,
                torch_vision_dataset=torch_vision_dataset if dataset_type == "torch_vision_dataset" else None,
            )
        except ValueError as e:
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 400
        else:
            return jsonify({
                "message": get_logs(),
                "project_info": Project_Info
            }), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

import torch.nn as nn

class MultiplyByOne(nn.Module):
    def forward(self, x):
        out = x * 2
        out = out + 0  # Adding a no-op to prevent graph simplification
        out = out / 2  # Adding a no-op to prevent graph simplification
        return out

import numpy as np
from finn.transformation.qonnx.qonnx_activation_handlers import QuantActBaseHandler, np_default_dtype
from qonnx.custom_op.registry import getCustomOp

class ReluToMultiThresholdHandler(QuantActBaseHandler):
    """Class for converting a regular non-quantized ReLU operation to a MultiThreshold node in Xilinx FINN."""
    def _check_compatibility(self):
        # Check if the ReLU operation is supported for transformation
        pass

    def _calculate_act_bias(self):
        # No bias is applied for standard ReLU
        return np.array([0.0], dtype=np_default_dtype)

    def _calculate_thresholds(self):
        # Thresholds for ReLU are typically defined as a single zero threshold
        output_shape = self._model.get_tensor_shape(self._q_node.output[0])
        num_output_channels = output_shape[1]  # Assuming NCHW or NHWC format
        thresholds = np.zeros((num_output_channels, 1), dtype=np_default_dtype)
        return thresholds

    def _calculate_act_scale(self):
        # Standard ReLU does not apply scaling
        return np.array([1.0], dtype=np_default_dtype)

    def _remove_activation_node(self, multi_threshold_node):
        # Find the activation node
        act_node = self._model.find_direct_predecessors(self._q_node)
        if act_node is None:
            raise RuntimeError(
                "For handling of Relu activations a predecesor to dgdgdg"
            )
        act_node = act_node[0]
        if act_node.op_type != "Relu":
            raise RuntimeError(
                "The predecesor of the Quant node must be Relu for handling "
                "of activations."
            )

        # Remove the activation node
        self._model.graph.node.remove(act_node)
        return multi_threshold_node

    def calculate_node_parameters(self):
        # Calculate all parameters for replacing ReLU with MultiThreshold
        return {
            "out_dtype": "FLOAT32",  # Regular ReLU typically works with float32
            "thresholds": self._calculate_thresholds(),
            "adder_bias": self._calculate_act_bias(),
            "mul_scale": self._calculate_act_scale(),
        }

    def replace_relu_node(self):
        """Replace a regular ReLU operation with a MultiThreshold node."""
        # Check compatibility
        self._check_compatibility()

        # Shorten instance variables
        model = self._model
        graph = model.graph
        n = self._q_node
        running_node_index = self._q_index

        # Calculate parameters for the MultiThreshold node
        parameter_dict = self.calculate_node_parameters()
        thresholds = parameter_dict["thresholds"]
        adder_bias = parameter_dict["adder_bias"]
        mul_scale = parameter_dict["mul_scale"]
        out_dtype = parameter_dict["out_dtype"]

        # Create threshold tensor
        thresh_tensor = oh.make_tensor_value_info(
            model.make_new_valueinfo_name(),
            TensorProto.FLOAT,
            thresholds.shape,
        )
        graph.value_info.append(thresh_tensor)
        model.set_initializer(thresh_tensor.name, thresholds)
        act_node = self._model.find_direct_predecessors(self._q_node)
        act_node = act_node[0]
        # Insert MultiThreshold node
        mt_node = oh.make_node(
            "MultiThreshold",
            [act_node.input[0], thresh_tensor.name],
            [n.output[0]],
            out_dtype=out_dtype,
            domain="qonnx.custom_op.general",
        )
        graph.node.insert(running_node_index, mt_node)
        running_node_index += 1

        # Get MultiThreshold instance
        mt_inst = getCustomOp(graph.node[running_node_index - 1])

        # Set bias and scale attributes
        mt_inst.set_nodeattr("out_scale", mul_scale[0].item())
        mt_inst.set_nodeattr("out_bias", adder_bias[0].item())
        mt_inst.set_nodeattr("out_dtype", out_dtype)

        # Remove the original ReLU node
        self._remove_activation_node(mt_node)
        graph.node.remove(n)

        # Return the updated model
        return self._model

from finn.transformation.qonnx.qonnx_activation_handlers import QuantReluHandler
from onnx import TensorProto
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from onnx import helper as oh

class CustomReLU(nn.Module):
    def forward(self, x):
        # Implementing an approximation of abs(x) using only arithmetic operations
        epsilon = 1e-6  # Small constant to avoid division by zero
        abs_x = (x * x + epsilon) / (x + epsilon)
        return (x + abs_x) / 2

class CustomReluBreakdown(Transformation):
    """
    Custom transformation to iterate through the nodes, identify ReLU nodes,
    and convert the next node to a quant node while applying the QuantReluHandler.
    """

    def __init__(self, bit_width=1, quant_type=DataType["UINT8"], narrow=0, signed=0):
        super().__init__()
        self.bit_width = bit_width
        self.quant_type = quant_type
        self.narrow = narrow
        self.signed = signed

    def apply(self, transform_model):
        graph = transform_model.graph
        graph_modified = False
        convert_next_node = False

        for idx, node in enumerate(graph.node):
            if convert_next_node:
                # Call QuantReluHandler on the current Quant node
                handler = ReluToMultiThresholdHandler(transform_model, node, idx)
                handler.replace_relu_node()

                convert_next_node = False
                graph_modified = True

            if node.op_type == "Relu":
                # Raise a flag to convert the next node to a quant node
                convert_next_node = True

       # transform_model = transform_model.transform(InferShapes())
        return (transform_model, graph_modified)

from onnx import helper as oh
from onnx import TensorProto
from qonnx.transformation.base import Transformation
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes
import numpy as np


class ReluToMultiThresholdTransform(Transformation):
    """
    Custom transformation to replace ReLU nodes with MultiThreshold nodes
    in Xilinx FINN. Works directly on ReLU nodes and reroutes their inputs/outputs.
    """

    def __init__(self):
        super().__init__()

    def apply(self, transform_model):
        graph = transform_model.graph
        graph_modified = False

        for idx, node in enumerate(graph.node):
            if node.op_type == "Relu":
                # Replace the ReLU node with a MultiThreshold node
                self.replace_relu_with_multithreshold(transform_model, node, idx)
                graph_modified = True

        return (transform_model, graph_modified)

    def replace_relu_with_multithreshold(self, transform_model, relu_node, relu_index):
        """
        Replaces a ReLU node with a MultiThreshold node.
        """
        graph = transform_model.graph

        # Get the input to the ReLU node (its predecessor)
        if len(relu_node.input) != 1:
            raise RuntimeError(f"ReLU node {relu_node.name} has unexpected inputs.")
        relu_input = relu_node.input[0]

        # Get the output of the ReLU node (its successor)
        if len(relu_node.output) != 1:
            raise RuntimeError(f"ReLU node {relu_node.name} has unexpected outputs.")
        relu_output = relu_node.output[0]

        # Calculate parameters for MultiThreshold
        thresholds = self._calculate_thresholds(transform_model, relu_node)
        adder_bias = self._calculate_act_bias()
        mul_scale = self._calculate_act_scale()
        out_dtype = "FLOAT32"

        # Create threshold tensor
        thresh_tensor = oh.make_tensor_value_info(
            transform_model.make_new_valueinfo_name(),
            TensorProto.FLOAT,
            thresholds.shape,
        )
        graph.value_info.append(thresh_tensor)
        transform_model.set_initializer(thresh_tensor.name, thresholds)

        # Create MultiThreshold node
        mt_node = oh.make_node(
            "MultiThreshold",
            inputs=[relu_input, thresh_tensor.name],
            outputs=[relu_output],
            domain="qonnx.custom_op.general",
        )
        mt_node.attribute.extend([
            oh.make_attribute("out_scale", float(mul_scale[0])),
            oh.make_attribute("out_bias", float(adder_bias[0])),
            oh.make_attribute("out_dtype", out_dtype),
        ])

        # Insert MultiThreshold node in place of ReLU node
        graph.node.insert(relu_index, mt_node)

        # Remove the original ReLU node
        graph.node.remove(relu_node)

    def _calculate_thresholds(self, transform_model, relu_node):
        """
        Calculates the thresholds for the MultiThreshold node based on the ReLU operation.
        """
        # ReLU threshold is always zero
        output_shape = transform_model.get_tensor_shape(relu_node.output[0])
        num_output_channels = output_shape[1]  # Assuming NCHW or NHWC format
        thresholds = np.zeros((num_output_channels, 1), dtype=np.float32)
        return thresholds

    def _calculate_act_bias(self):
        """
        Calculates the bias for the MultiThreshold node.
        """
        # ReLU does not apply a bias
        return np.array([0.0], dtype=np.float32)

    def _calculate_act_scale(self):
        """
        Calculates the scale for the MultiThreshold node.
        """
        # ReLU does not apply scaling
        return np.array([1.0], dtype=np.float32)


import traceback
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.general import ConvertSubToAdd, ConvertDivToMul
from finn.transformation.streamline import Streamline, RoundAndClipThresholds, CollapseRepeatedMul, ConvertSignToThres, \
    CollapseRepeatedAdd
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.streamline.absorb import (AbsorbTransposeIntoMultiThreshold,
                                                   AbsorbScalarMulAddIntoTopK,
                                                   AbsorbSignBiasIntoMultiThreshold,
                                                   AbsorbAddIntoMultiThreshold,
                                                   AbsorbMulIntoMultiThreshold, FactorOutMulSignMagnitude,
                                                   Absorb1BitMulIntoMatMul, Absorb1BitMulIntoConv)
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants, MoveAddPastMul, \
    MoveScalarAddPastMatMul, MoveAddPastConv, MoveScalarMulPastMatMul, MoveScalarMulPastConv, \
    MoveMaxPoolPastMultiThreshold, MoveLinearPastEltwiseAdd, MoveLinearPastFork
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes


def streamline_transforms(input_streamline_model, save_name):
    """
    Applies a series of streamlining transformations to a model and saves the resulting model.
    If a transformation fails, it logs the error and moves on to the next transformation.

    Parameters:
        input_streamline_model (ModelWrapper): The model to transform.
        save_name (str): The path to save the transformed model.

    Returns:
        ModelWrapper: The transformed model.
    """
    transformations = [
        AbsorbSignBiasIntoMultiThreshold(),
        ConvertSubToAdd(),
        ConvertDivToMul(),
        CollapseRepeatedMul(),
        BatchNormToAffine(),
        ConvertSignToThres(),
        MoveAddPastMul(),
        MoveScalarAddPastMatMul(),
        MoveAddPastConv(),
        MoveScalarMulPastMatMul(),
        MoveScalarMulPastConv(),
        MoveAddPastMul(),
        MoveScalarLinearPastInvariants(),
        CollapseRepeatedAdd(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        MoveMaxPoolPastMultiThreshold(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(),
        Absorb1BitMulIntoConv(),
        AbsorbMulIntoMultiThreshold(),
        Streamline(),
        LowerConvsToMatMul(),
        MakeMaxPoolNHWC(),
        AbsorbTransposeIntoMultiThreshold(),
        ConvertBipolarMatMulToXnorPopcount(),
        Streamline(),
        AbsorbScalarMulAddIntoTopK(),
        RoundAndClipThresholds(),
        MoveLinearPastEltwiseAdd(),
        MoveLinearPastFork()
    ]

    input_streamline_model = input_streamline_model.transform(ReluToMultiThresholdTransform())
    input_streamline_model = tidy_up_transforms(input_streamline_model, save_name)
    for iter_id in range(3):
        for transform in transformations:
            try:
                input_streamline_model = input_streamline_model.transform(transform)
            except Exception as e:
                print(e)

        input_streamline_model = tidy_up_transforms(input_streamline_model, save_name)

        input_streamline_model = input_streamline_model.transform(InferDataLayouts())
        input_streamline_model = input_streamline_model.transform(RemoveUnusedTensors())
        input_streamline_model = input_streamline_model.transform(DoubleToSingleFloat())
        input_streamline_model = input_streamline_model.transform(SortGraph())
        input_streamline_model = input_streamline_model.transform(RemoveIdentityOps())

    return input_streamline_model


from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferBinaryMatrixVectorActivation,
    InferQuantizedMatrixVectorActivation,
    InferLabelSelectLayer,
    InferThresholdingLayer,
    InferStreamingMaxPool,
    InferConvInpGen,
    InferAddStreamsLayer,
    InferChannelwiseLinearLayer,
    InferConcatLayer,
    InferDuplicateStreamsLayer,
    InferGlobalAccPoolLayer,
    InferLookupLayer,
    InferPool,
    InferStreamingEltwise,
    InferUpsample,
    InferVectorVectorActivation
)


def to_hw_transforms(input_hw_model, save_name):
    """
    Applies a comprehensive series of hardware-oriented transformations to a model and saves the resulting model.
    If a transformation fails, it logs the error and moves on to the next transformation.

    Parameters:
        input_hw_model (ModelWrapper): The model to transform.
        save_name (str): The path to save the transformed model.

    Returns:
        ModelWrapper: The transformed model.
    """
    transformations = [
        InferBinaryMatrixVectorActivation(),
        InferQuantizedMatrixVectorActivation(),
        InferLabelSelectLayer(),
        InferThresholdingLayer(),
        InferConvInpGen(),
        InferStreamingMaxPool(),
        InferAddStreamsLayer(),
        InferChannelwiseLinearLayer(),
        InferConcatLayer(),
        InferDuplicateStreamsLayer(),
        InferGlobalAccPoolLayer(),
        InferLookupLayer(),
        InferPool(),
        InferStreamingEltwise(),
        InferUpsample(),
        InferVectorVectorActivation(),
        RemoveCNVtoFCFlatten(),
        AbsorbConsecutiveTransposes()
    ]

    for iter_id in range(3):
        for transform in transformations:
            try:
                input_hw_model = input_hw_model.transform(transform)
            except Exception as e:
                print(e)
                pass

        # Apply final tidy-up transformations
        input_hw_model = tidy_up_transforms(input_hw_model, save_name)

        input_hw_model = input_hw_model.transform(InferDataLayouts())
        input_hw_model = input_hw_model.transform(RemoveUnusedTensors())
        input_hw_model = input_hw_model.transform(DoubleToSingleFloat())
        input_hw_model = input_hw_model.transform(SortGraph())
        input_hw_model = input_hw_model.transform(RemoveIdentityOps())

    # Save the final transformed model
    input_hw_model.save(save_name)

    return input_hw_model


from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)


def dataflow_partitioning(input_data_model, save_name):
    """
    Applies dataflow partitioning transformation to the model and saves the resulting parent and dataflow models.

    Parameters:
        input_data_model (ModelWrapper): The model to transform.
        save_name (str): The directory to save the transformed models.

    Returns:
        ModelWrapper: The transformed parent model.
    """
    # Apply dataflow partitioning
    parent_model = input_data_model.transform(CreateDataflowPartition())
    parent_model.save(save_name)

    # Retrieve the dataflow partition model filename
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    # Load and return the dataflow model
    dataflow_model = ModelWrapper(dataflow_model_filename)

    return dataflow_model


from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def specialize_layers_transform(input_specialize_model, board_name, save_name):
    """
    Applies layer specialization transformation to a dataflow model for the specified FPGA part and saves the resulting model.

    Parameters:
        input_specialize_model (ModelWrapper): The dataflow model to transform.
        board_name (str): The FPGA board for which to specialize the layers.
        save_name (str): The path to save the specialized model.

    Returns:
        ModelWrapper: The transformed and specialized dataflow model.
    """
    fpga_part = pynq_part_map[board_name]
    # Apply specialization for FPGA layers
    input_specialize_model = input_specialize_model.transform(SpecializeLayers(fpga_part))

    # Save the specialized model
    input_specialize_model.save(save_name)

    return input_specialize_model


def folding_transform(input_folding_model, save_name):
    """
    Applies folding configuration to fully connected (MVAU_hls) and sliding window (ConvolutionInputGenerator_rtl)
    layers in the model and saves the resulting model.

    Parameters:
        input_folding_model (ModelWrapper): The specialized model to transform.
        save_name (str): Directory to save the transformed model.

    Returns:
        ModelWrapper: The transformed and folded dataflow model.
    """
    folding_config = [
        (16, 3, [128]),
        (32, 32, [128]),
        (16, 32, [128]),
        (16, 32, [128]),
        (4, 32, [81]),
        (1, 32, [2]),
        (1, 4, [2]),
        (1, 8, [128]),
        (5, 1, [3]),
    ]

    # Apply folding configuration to fully connected layers
    fc_layers = input_folding_model.get_nodes_by_op_type("MVAU_hls")
    for fcl, (pe, simd, ififodepth) in zip(fc_layers, folding_config):
        fcl_inst = getCustomOp(fcl)
        fcl_inst.set_nodeattr("PE", pe)
        fcl_inst.set_nodeattr("SIMD", simd)
        fcl_inst.set_nodeattr("inFIFODepths", ififodepth)

    # Apply SIMD values from the folding configuration to sliding window layers
    swg_layers = input_folding_model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")
    for i in range(len(swg_layers)):
        swg_inst = getCustomOp(swg_layers[i])
        simd = folding_config[i][1]
        swg_inst.set_nodeattr("SIMD", simd)

    # Apply unique node names to all nodes and save the transformed model
    input_folding_model = input_folding_model.transform(GiveUniqueNodeNames())
    input_folding_model.save(save_name)

    return input_folding_model


from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild


def zynq_build_transform(input_zynq_model, save_name, brd_name):
    """
    Applies the ZynqBuild transformation to a model for the specified PYNQ board and clock period.

    Parameters:
        input_zynq_model (ModelWrapper): Folded model to transform.
        save_name (str): Directory to save the transformed model.
        brd_name (str): Name of the PYNQ board to target.

    Returns:
        ModelWrapper: The transformed model after applying ZynqBuild.
    """
    target_clk_ns = 10
    # Apply ZynqBuild transformation
    input_zynq_model = input_zynq_model.transform(ZynqBuild(platform=brd_name, period_ns=target_clk_ns))

    # Save the transformed model
    input_zynq_model.save(save_name)

    return input_zynq_model


from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver


def pynq_driver_transform(input_driver_model, save_name):
    """
    Applies the MakePYNQDriver transformation to the model to generate a PYNQ-compatible driver.

    Parameters:
        input_driver_model (ModelWrapper): ZynqBuild model to transform.
        save_name (str): Directory to save the transformed model.

    Returns:
        ModelWrapper: The transformed model with PYNQ driver compatibility.
    """
    # Apply MakePYNQDriver transformation
    input_driver_model = input_driver_model.transform(MakePYNQDriver("zynq-iodma"))

    # Save the transformed model
    input_driver_model.save(save_name)

    return input_driver_model



@app.route('/getupdate', methods=['GET'])
def get_update():
    """
    Provide incremental updates for the progress of the setup process.
    """
    global progress_step
    global input_model
    global input_model_shape
    global export_onnx_path
    global model
    try:
        if progress_step == "Done":
            final_message = get_logs() + "<br><b>[INFO]</b>  All steps completed. Setup process finalized."
            print(f"{final_message}")
            return jsonify({"step": progress_step, "message": final_message, "log": f"{final_message}"}), 200
        if progress_step == "Project Setup":
            if Project_Info['Model_Type'] == "untrained":
                log_message("Training for untrained models are not supported at the moment!", level="error")
            elif Project_Info['Model_Type'] == "custom_pretrained":
                log_message("Custom Pretrained models are not supported at the moment!", level="error")
            elif Project_Info['Model_Type'] == "sample_pretrained" or Project_Info['Model_Type'] == "finn_pretrained":
                pretrained_folder = os.path.join(Project_Info['Folder'], "src")
                input_model, input_model_shape = load_pretrained_model(Project_Info['Pretrained_Model'],
                                                                       Project_Info['Model_Type'], pretrained_folder,
                                                                       initial_channels=1)
            else:
                log_message("Unsupported Model Type", level="error")
            progress_step = "Export Qonnx"
            return jsonify({"step": progress_step, "message": get_logs(), "log": f"{get_logs()}"}), 200

        if progress_step == "Export Qonnx":
            from brevitas.export import export_qonnx
            from qonnx.util.cleanup import cleanup as qonnx_cleanup

            export_onnx_path = set_onnx_checkpoint(Project_Info, "Brevitas Export")
            export_qonnx(input_model, torch.randn(input_model_shape), export_onnx_path, opset_version=9)
            qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)

            progress_step = "Convert to Finn"
            return jsonify({"step": progress_step, "message": get_logs(), "log": f"{get_logs()}"}), 200

        if progress_step == "Convert to Finn":
            from qonnx.core.modelwrapper import ModelWrapper
            from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

            model = ModelWrapper(export_onnx_path)
            model = model.transform(ConvertQONNXtoFINN())
            model.save(set_onnx_checkpoint(Project_Info, "QONNX to FINN"))

            progress_step = "Initial Tidying Up"
            return jsonify({"step": progress_step, "message": get_logs(), "log": f"{get_logs()}"}), 200

        if progress_step == "Initial Tidying Up":
            model = tidy_up_transforms(model, set_onnx_checkpoint(Project_Info, "Tidy ONNX Post Finn"))
            progress_step = "PostPre Processing"
            return jsonify({"step": progress_step, "message": get_logs(), "log": f"{get_logs()}"}), 200

        if progress_step == "PostPre Processing":
            from qonnx.transformation.insert_topk import InsertTopK
            global_inp_name = model.graph.input[0].name
            ishape = model.get_tensor_shape(global_inp_name)
            mul_model = MultiplyByOne()
            from brevitas.export import export_qonnx
            chkpt_mul_name = set_onnx_checkpoint(Project_Info, "Mul_One_ONNX")
            export_qonnx(mul_model, torch.randn(ishape), chkpt_mul_name, opset_version=9)
            from qonnx.util.cleanup import cleanup as qonnx_cleanup
            qonnx_cleanup(chkpt_mul_name, out_file=chkpt_mul_name)
            from qonnx.core.modelwrapper import ModelWrapper

            from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
            mul_model_wrapper = ModelWrapper(chkpt_mul_name)
            mul_model_wrapper = mul_model_wrapper.transform(ConvertQONNXtoFINN())
            model = model.transform(MergeONNXModels(mul_model_wrapper))

            global_inp_name = model.graph.input[0].name
            model.set_tensor_datatype(global_inp_name, DataType["UINT8"])

            model = model.transform(InsertTopK(k=1))
            model.save(set_onnx_checkpoint(Project_Info, "Post Processing"))
            model = tidy_up_transforms(model, set_onnx_checkpoint(Project_Info, "Tidy Post PrePost Proc"))

            progress_step = "Streamlining"
            return jsonify({"step": progress_step, "message": get_logs(), "log": f"{get_logs()}"}), 200

        if progress_step == "Streamlining":
            model = streamline_transforms(model, set_onnx_checkpoint(Project_Info, "Streamlined ONNX"))
            progress_step = "HW Layers"
            return jsonify({"step": progress_step, "message": get_logs(), "log": f"{get_logs()}"}), 200

        if progress_step == "HW Layers":
            model = to_hw_transforms(model, set_onnx_checkpoint(Project_Info, "To HW Layers"))
            progress_step = "Dataflow Partitioning"
            return jsonify({"step": progress_step, "message": get_logs(), "log": f"{get_logs()}"}), 200

        if progress_step == "Dataflow Partitioning":
            model = dataflow_partitioning(model, set_onnx_checkpoint(Project_Info, "Dataflow Partition Parent Model"))
            model.save(set_onnx_checkpoint(Project_Info, "Dataflow Partition Streaming Model"))
            progress_step = "Specialize Layers"
            return jsonify({"step": progress_step, "message": get_logs(), "log": f"{get_logs()}"}), 200


        if progress_step == "Specialize Layers":
            model = specialize_layers_transform(model, Project_Info['Board_name'], set_onnx_checkpoint(Project_Info,
                                                                                                       f"Specialize Model Layers to {Project_Info['Board_name']}"))
            progress_step = "ZYNQ Build"
            return jsonify({"step": progress_step, "message": get_logs(), "log": f"{get_logs()}"}), 200

        return jsonify({"step": progress_step, "message": get_logs(), "log": f"{get_logs()}"}), 200

    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(f"[ERROR] {error_message}")
        return jsonify({"error": error_message}), 500
