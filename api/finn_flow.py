from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths

from helper import *

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
    log_message("Setting up project")
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
        project_folder = f"{WORKING_FOLDER}{prj_name_stripped}_{timestamp}"

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
        if not sample_pretrained_model or sample_pretrained_model not in sample_pretrained_models:
            log_message(f"Sample Pretrained Model must be one of: {sample_pretrained_models}", level="error")
        prj_info["Pretrained_Model"] = sample_pretrained_model

    if model_type == "sample_untrained":
        if not sample_untrained_model or sample_untrained_model not in sample_untrained_models:
            log_message(f"Sample Untrained Model must be one of: {sample_untrained_models}", level="error")
        prj_info["Sample_Untrained_Model"] = sample_untrained_model

    if model_type == "finn_pretrained":
        if not finn_pretrained_model or finn_pretrained_model not in finn_pretrained_models:
            log_message(f"Finn Pretrained Model must be one of: {finn_pretrained_models}", level="error")
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
            if not torch_vision_dataset or torch_vision_dataset.lower() not in available_datasets:
                log_message(f"Torch Vision Dataset must be one of: {available_datasets}", level="error")
            prj_info["Torch_Vision_Dataset"] = torch_vision_dataset

    log_message(f"Project setup complete. {prj_name} has been initialized.")
    return prj_info


from finn.util.test import get_test_model_trained
import torch

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
        if model_name == "alexnet_3w3a_cifar10":
            pretrained_model = QuantAlexNet(num_bits=3, num_classes=10).to('cpu')
            pretrained_model.load_state_dict(
                torch.load("/home/fastqnn/finn/notebooks/Fast-QNN/alexnet_3w3a_cifar10.pth",
                           map_location=torch.device('cpu')))
            pretrained_model.eval()
        if model_name == "alexnet_3w3a_mnist":
            pretrained_model = QuantAlexNet(num_bits=3, num_classes=10).to('cpu')
            pretrained_model.load_state_dict(torch.load("/home/fastqnn/finn/notebooks/Fast-QNN/alexnet_3w3a_mnist.pth",
                                                        map_location=torch.device('cpu')))
            pretrained_model.eval()
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

from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup

from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

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

from qonnx.core.datatype import DataType
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from finn.util.pytorch import ToTensor

import torch.nn as nn

class MultiplyByOne(nn.Module):
    def forward(self, x):
        out = x * 2
        out = out + 0  # Adding a no-op to prevent graph simplification
        out = out / 2  # Adding a no-op to prevent graph simplification
        return out
from qonnx.transformation.insert_topk import InsertTopK

import numpy as np
from qonnx.custom_op.registry import getCustomOp

from onnx import helper as oh
from onnx import TensorProto
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes


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

import math
import os
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from qonnx.custom_op.registry import getCustomOp
from functools import partial
import matplotlib.pyplot as plt

def get_int_attr(node_inst, attr_name = None):
    """
    Safely retrieves an integer attribute from a node.

    Parameters:
        node_inst: The node instance.
        attr_name (str): The name of the attribute to retrieve.

    Returns:
        int: The integer value of the attribute.
    """
    if attr_name is not None:
        attr = node_inst.get_nodeattr(attr_name)
    else:
        attr = node_inst
    if isinstance(attr, (tuple, list)):
        if len(attr) == 0:
            raise ValueError(f"Attribute '{attr_name}' for node '{node_inst.name}' is an empty tuple/list.")
        return get_int_attr(attr[0])
    elif isinstance(attr, int):
        return attr
    else:
        raise TypeError(f"Attribute '{attr_name}' for node '{node_inst.name}' is of unsupported type {type(attr)}.")


from qonnx.custom_op.registry import getCustomOp

def calculate_simd(ifm_channels, max_simd):
    """
    Calculates the largest possible SIMD value that divides IFMChannels without remainder.

    Parameters:
        ifm_channels (int): Number of Input Feature Map Channels.
        max_simd (int): Maximum allowed SIMD value.

    Returns:
        int: Calculated SIMD value.
    """
    for simd in range(max_simd, 0, -1):
        if ifm_channels % simd == 0:
            return simd
    return 1  # Fallback to 1 if no divisors found


def generate_valid_folding_factors(
    folding_model,
    max_pe=8,         # Significantly reduced
    max_simd=8,       # Significantly reduced
    max_bitwidth=8191,
    max_total_ram=280   # FPGA RAMB18 capacity
):
    """
    Generates a folding configuration dictionary ensuring all constraints are met,
    utilizing the 'infifodepth' attributes set by InsertAndSetFIFODepths.

    Parameters:
        folding_model (ModelWrapper): The FINN model.
        max_pe (int): Maximum allowed PE value.
        max_simd (int): Maximum allowed SIMD value.
        max_bitwidth (int): Maximum allowed stream width in bits.
        max_total_ram (int): Maximum total RAMB18s available on the FPGA.

    Returns:
        dict: A dictionary with node names as keys and their folding parameters as values.
    """
    folding_config = {}

    relevant_node_types = ["MVAU_hls", "VVAU", "ConvolutionInputGenerator_rtl"]

    for node_type in relevant_node_types:
        layers = folding_model.get_nodes_by_op_type(node_type)
        for node in layers:
            node_inst = getCustomOp(node)
            node_name = node.name

            if node_type == "ConvolutionInputGenerator_rtl":
                # For ConvolutionInputGenerator_rtl, set only SIMD
                # Derive SIMD based on IFMChannels
                ifm_channels = node_inst.get_nodeattr("IFMChannels")  # Assuming NHWC or similar format
                simd = calculate_simd(ifm_channels, max_simd)

                # Validate bitwidth constraint
                input_bitwidth = node_inst.get_input_datatype().bitwidth()
                assert simd * input_bitwidth <= max_bitwidth, f"SIMD * input_bitwidth for node '{node_name}' exceeds max_bitwidth"

                folding_config[node_name] = {
                    "SIMD": simd
                }
                log_message(f"Node '{node_name}': Set SIMD={simd} based on IFMChannels={ifm_channels}.", "info")
                continue
            try:
                MW = get_int_attr(node_inst, "MW")
                MH = get_int_attr(node_inst, "MH")
            except (ValueError, TypeError, AttributeError) as e:
                log_message(f"Error retrieving attributes for node '{node_name}': {e}", "error")
                continue  # Skip this node or handle as needed

            input_bitwidth = node_inst.get_input_datatype().bitwidth()
            output_bitwidth = node_inst.get_output_datatype().bitwidth()

            # Calculate minimum required SIMD based on HLS constraint
            min_simd = math.ceil(MW / 1024)
            log_message(f"Node '{node_name}': MW={MW}, min_simd={min_simd}", "info")

            # Determine valid SIMD: largest divisor of MW <= max_simd and ensures (SIMD * input_bitwidth) <= max_bitwidth
            simd_candidates = [
                x for x in range(min_simd, min(max_simd, MW) + 1)
                if MW % x == 0 and (x * input_bitwidth) <= max_bitwidth
            ]
            simd = max(simd_candidates) if simd_candidates else min_simd
            log_message(f"Node '{node_name}': SIMD candidates={simd_candidates}, selected SIMD={simd}", "info")

            # Determine valid PE: largest divisor of MH <= max_pe and ensures (PE * output_bitwidth) <= max_bitwidth
            pe_candidates = [
                x for x in range(1, min(max_pe, MH) + 1)
                if MH % x == 0 and (x * output_bitwidth) <= max_bitwidth
            ]
            pe = max(pe_candidates) if pe_candidates else 1
            log_message(f"Node '{node_name}': PE candidates={pe_candidates}, selected PE={pe}", "info")

            folding_config[node_name] = {
                "PE": pe,
                "SIMD": simd
            }
            log_message(
                f"Configured node '{node_name}' with PE={pe}, SIMD={simd}. "
                "info"
            )

    return folding_config

def apply_folding_config(folding_model, folding_config):
    """
    Applies the folding configuration to the model.

    Parameters:
        folding_model (ModelWrapper): The FINN model.
        folding_config (dict): Folding configuration dictionary.

    Returns:
        ModelWrapper: The model with updated folding parameters.
    """
    for node_name, config in folding_config.items():
        try:
            # Attempt to retrieve the node by its name
            node = folding_model.get_node_from_name(node_name)
        except AttributeError:
            log_message(f"Node '{node_name}' not found in the model.", "error")
            continue  # Skip to the next node if not found

        # Retrieve the custom operation instance
        node_inst = getCustomOp(node)

        # Flag to track if any attribute was set for logging purposes
        attributes_set = []

        # Assign PE if it exists in the configuration
        if "PE" in config:
            pe_value = config["PE"]
            node_inst.set_nodeattr("PE", pe_value)
            attributes_set.append(f"PE={pe_value}")
            log_message(f"Node '{node_name}': Set PE to {pe_value}.", "info")
        else:
            log_message(f"Node '{node_name}': 'PE' not specified in configuration. Skipping PE assignment.", "debug")

        # Assign SIMD if it exists in the configuration
        if "SIMD" in config:
            simd_value = config["SIMD"]
            node_inst.set_nodeattr("SIMD", simd_value)
            attributes_set.append(f"SIMD={simd_value}")
            log_message(f"Node '{node_name}': Set SIMD to {simd_value}.", "info")
        else:
            log_message(f"Node '{node_name}': 'SIMD' not specified in configuration. Skipping SIMD assignment.", "debug")

        # Log summary of assignments for the current node
        if attributes_set:
            assigned_attrs = ", ".join(attributes_set)
            log_message(f"Node '{node_name}': Assigned attributes -> {assigned_attrs}.", "info")
        else:
            log_message(f"Node '{node_name}': No attributes assigned.", "warning")

    return folding_model


def validate_stream_widths(folding_model, max_bitwidth=8191):
    """
    Validates that all stream widths are within the allowed maximum.

    Parameters:
        folding_model (ModelWrapper): The FINN model.
        max_bitwidth (int): Maximum allowed stream width in bits.

    Raises:
        AssertionError: If any stream width exceeds max_bitwidth.
    """
    # Check MVAU_hls layers
    fc_layers = folding_model.get_nodes_by_op_type("MVAU_hls")
    for node in fc_layers:
        node_inst = getCustomOp(node)
        simd = node_inst.get_nodeattr("SIMD")
        pe = node_inst.get_nodeattr("PE")
        input_bitwidth = node_inst.get_input_datatype().bitwidth()
        output_bitwidth = node_inst.get_output_datatype().bitwidth()

        instream_width = simd * input_bitwidth
        outstream_width = pe * output_bitwidth

        assert instream_width <= max_bitwidth, f"{node.name} has input stream width {instream_width} bits > {max_bitwidth}"
        assert outstream_width <= max_bitwidth, f"{node.name} has output stream width {outstream_width} bits > {max_bitwidth}"

    # Check VVAU layers if any
    vvau_layers = folding_model.get_nodes_by_op_type("VVAU")
    for node in vvau_layers:
        node_inst = getCustomOp(node)
        simd = node_inst.get_nodeattr("SIMD")
        pe = node_inst.get_nodeattr("PE")
        input_bitwidth = node_inst.get_input_datatype().bitwidth()
        output_bitwidth = node_inst.get_output_datatype().bitwidth()

        instream_width = simd * input_bitwidth
        outstream_width = pe * output_bitwidth

        assert instream_width <= max_bitwidth, f"{node.name} has input stream width {instream_width} bits > {max_bitwidth}"
        assert outstream_width <= max_bitwidth, f"{node.name} has output stream width {outstream_width} bits > {max_bitwidth}"

    log_message("All stream widths are within the allowed maximum.", "info")

def insert_and_set_fifo_depths(folding_model, fpgapart, clk_ns=10.0, max_qsrl_depth=256, max_depth=None, swg_exception=False, vivado_ram_style='auto', force_python_sim=False):
    """
    Inserts and sets FIFO depths using the InsertAndSetFIFODepths transformation.

    Parameters:
        folding_model (ModelWrapper): The FINN model.
        fpgapart (str): FPGA part identifier.
        clk_ns (float): Clock period in nanoseconds.
        max_qsrl_depth (int): Threshold to use Vivado IP for FIFO depths exceeding this value.
        max_depth (int or None): Initial depth for the largest FIFOs.
        swg_exception (bool): Adjust convolution FIFO depths if True.
        vivado_ram_style (str): RAM style attribute for Vivado-implemented FIFOs.
        force_python_sim (bool): Force the use of Python-based simulation.

    Returns:
        ModelWrapper: The model with inserted and set FIFO depths.
    """
    insert_set_fifo_depths = InsertAndSetFIFODepths(
        fpgapart=fpgapart,
        clk_ns=clk_ns,
        max_qsrl_depth=max_qsrl_depth,
        max_depth=max_depth,
        swg_exception=swg_exception,
        vivado_ram_style=vivado_ram_style,
        force_python_sim=force_python_sim
    )
    folding_model = folding_model.transform(insert_set_fifo_depths)
    log_message("Inserted and set FIFO depths using InsertAndSetFIFODepths.", "info")
    return folding_model

def insert_dwcs_if_needed(model_dwc):
    """
    Inserts DataWidthConverters (DWCs) where stream widths between consecutive layers do not match.

    Parameters:
        model_dwc (ModelWrapper): The FINN model.

    Returns:
        ModelWrapper: The model with DWCs inserted where necessary.
    """

    # InsertDWC handles the insertion of DWCs automatically when stream widths mismatch.
    model_dwc = model_dwc.transform(InsertDWC())
    log_message("Inserted DataWidthConverters where necessary.", "info")

    return model_dwc

def folding_transform(input_folding_model, board_name, save_name):
    """
    Applies a user-defined folding configuration to MVAU_hls and VVAU layers, inserts and sets FIFO depths,
    inserts DWCs, validates the model, and saves the result.

    Parameters:
        input_folding_model (ModelWrapper): The specialized model to transform.
        save_name (str): Path to save the transformed model (e.g., checkpoint).
        fpgapart (str): FPGA part identifier.

    Returns:
        ModelWrapper: The transformed and folded dataflow model.
    """
    # Set default parameters or override with kwargs
    max_pe = 1
    max_simd = 1
    max_bitwidth = 8191
    max_total_ram = 280
    fpgapart = pynq_part_map[board_name]

    # Apply unique node naming transformation
    input_folding_model = input_folding_model.transform(GiveUniqueNodeNames())
    log_message("Applied GiveUniqueNodeNames transformation.", "info")

    folding_config = generate_valid_folding_factors(
        input_folding_model,
        max_pe=max_pe,
        max_simd=max_simd,
        max_bitwidth=max_bitwidth,
        max_total_ram=max_total_ram
    )

    # Apply folding configuration
    input_folding_model = apply_folding_config(input_folding_model, folding_config)
    ''' 
    # Insert and set FIFO depths first
    input_folding_model = insert_and_set_fifo_depths(
        input_folding_model,
        fpgapart=fpgapart,
        clk_ns=10.0,
        max_qsrl_depth=256,
        max_depth=None,  # Use tensor size as initial depth
        swg_exception=False,
        vivado_ram_style='auto',
        force_python_sim=False
    )
    '''

    log_message("Applied folding configuration to the model.", "info")

    # Validate stream widths before inserting DWCs
    try:
        validate_stream_widths(input_folding_model, max_bitwidth=max_bitwidth)
    except AssertionError as e:
        log_message(f"Stream width validation failed: {e}", "error")
        # Implement a fallback strategy: reduce max_simd and max_pe by half and retry
        log_message("Reducing max_simd and max_pe by half and retrying...", "warning")
        new_max_simd = max(1, max_simd // 2)
        new_max_pe = max(1, max_pe // 2)
        folding_config = generate_valid_folding_factors(
            input_folding_model,
            max_pe=new_max_pe,
            max_simd=new_max_simd,
            max_bitwidth=max_bitwidth,
            max_total_ram=max_total_ram
        )
        input_folding_model = apply_folding_config(input_folding_model, folding_config)
        # Re-validate
        validate_stream_widths(input_folding_model, max_bitwidth=max_bitwidth)
        log_message("Re-validated stream widths after fallback strategy.", "info")

    # Insert DWCs where needed
    # input_folding_model = insert_dwcs_if_needed(input_folding_model)

    # Save the folded model
    input_folding_model.save(save_name)
    log_message(f"Folded model saved to '{save_name}'.", "info")

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

from shutil import copy
from distutils.dir_util import copy_tree
from finn.util.basic import make_build_dir
from shutil import make_archive
