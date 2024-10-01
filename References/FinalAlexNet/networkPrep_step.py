import torch
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants


# Load the FINN-ONNX model (output of the previous step)
model = ModelWrapper('outputs/alexnet_1w1a_mnist_finn.onnx')

model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(RemoveStaticGraphInputs())

# Save the tidied-up model
model.save('outputs/alexnet_1w1a_mnist_tidy.onnx')
print("Model tidied up and saved as 'alexnet_1w1a_mnist_tidy.onnx'")

from finn.util.pytorch import ToTensor
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.core.datatype import DataType
import brevitas.onnx as bo
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.infer_datatypes import InferDataTypes

# Load the model and get the input shape
model = ModelWrapper('outputs/alexnet_1w1a_mnist_tidy.onnx')
global_inp_name = model.graph.input[0].name
ishape = model.get_tensor_shape(global_inp_name)

# Create a dummy tensor with the given input shape
dummy_input = torch.randn(ishape)  # Create a tensor based on the shape

# Apply ToTensor transformation manually
totensor_pyt = ToTensor()
processed_input = totensor_pyt.forward(dummy_input)

# Export the ToTensor operation (QONNX expects actual tensor, not shape)
chkpt_preproc_name = "outputs/alexnet_1w1a_mnist_preproc.onnx"
bo.export_qonnx(totensor_pyt, processed_input, chkpt_preproc_name, opset_version=9)  # Pass processed_input, not shape

# join preprocessing and core model
pre_model = ModelWrapper(chkpt_preproc_name)
model = model.transform(MergeONNXModels(pre_model))
# add input quantization annotation: UINT8 for all BNN-PYNQ models
global_inp_name = model.graph.input[0].name
model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
model = model.transform(InsertTopK(k=1))
chkpt_name = "outputs/alexnet_1w1a_mnist_pre_post.onnx"
# tidy-up again
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferDataTypes())
model = model.transform(RemoveStaticGraphInputs())
model.save(chkpt_name)
print("Model Tidied again after Pre-Post processing and saved as 'alexnet_1w1a_mnist_pre_post.onnx'")

from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.general import RemoveUnusedTensors

model = ModelWrapper('outputs/alexnet_1w1a_mnist_pre_post.onnx')

# Apply streamlining transformations
model = model.transform(MoveScalarLinearPastInvariants())
model = model.transform(LowerConvsToMatMul())
model = model.transform(MakeMaxPoolNHWC())
model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())

# Save the streamlined model
model.save('outputs/alexnet_1w1a_mnist_streamlined.onnx')
print("Model streamlined and saved as 'alexnet_1w1a_mnist_streamlined.onnx'")

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_data_layouts import InferDataLayouts

mem_mode = "decoupled"
model = ModelWrapper('outputs/alexnet_1w1a_mnist_streamlined.onnx')

model = model.transform(to_hls.InferBinaryMatrixVectorActivation())
model = model.transform(to_hls.InferQuantizedMatrixVectorActivation())
# TopK to LabelSelect
model = model.transform(to_hls.InferLabelSelectLayer())
# input quantization (if any) to standalone thresholding
model = model.transform(to_hls.InferThresholdingLayer())
model = model.transform(to_hls.InferStreamingMaxPool())
# get rid of Reshape(-1, 1) operation between hlslib nodes
model = model.transform(RemoveCNVtoFCFlatten())
# get rid of Tranpose -> Tranpose identity seq
model = model.transform(absorb.AbsorbConsecutiveTransposes())
model = model.transform(InferDataLayouts())
# Save the model with HW abstraction layers

from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.set_folding import SetFolding


# Apply dataflow partitioning
# model = model.transform(CreateDataflowPartition())

model.save('outputs/alexnet_1w1a_mnist_hw.onnx')
print("Model with HW layers saved as 'alexnet_1w1a_mnist_hw.onnx'")