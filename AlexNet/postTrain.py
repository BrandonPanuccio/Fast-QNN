import brevitas.onnx as bo
import onnx
import torch
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes

from SimpleAlexNet import SimpleAlexNet

buildName = "alexnet"
# Load the trained model
cnv = SimpleAlexNet()
cnv.load_state_dict(torch.load("outputs/"+buildName+".pth", map_location=torch.device('cpu')))
cnv.eval()

# Export the model using brevitas
input_tensor = torch.randn(1, 1, 32, 32)  # Example input tensor with 32x32 dimensions for AlexNet-like model
bo.export_qonnx(cnv, input_tensor, "outputs/"+buildName+"_export.onnx")


torch.onnx.export(cnv, input_tensor, "outputs/"+buildName+"_v9.onnx", opset_version=9)

model = ModelWrapper("outputs/"+buildName+"_v9.onnx")

from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition

test_pynq_board = "Pynq-Z2"
target_clk_ns = 10

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
# First, partition the model into streaming dataflow
model = model.transform(CreateDataflowPartition())
# Apply shape inference to ensure all tensor shapes are known
model = model.transform(InferShapes())
# Apply the PYNQ Driver transformation
model = model.transform(ConvertQONNXtoFINN())

# Save the final model
model.save("outputs/"+buildName+"_synth.onnx")
