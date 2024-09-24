import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul

# Apply transformations
model = ModelWrapper("outputs/simple_nn_pre_post.onnx")
model = model.transform(MoveScalarLinearPastInvariants())
model = model.transform(Streamline())
model = model.transform(LowerConvsToMatMul())
model = model.transform(MakeMaxPoolNHWC())
model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
model = model.transform(ConvertBipolarMatMulToXnorPopcount())
model = model.transform(Streamline())
model = model.transform(MakeMaxPoolNHWC())
model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())
model.save("outputs/simple_nn_streamlined.onnx")

# Choose the memory mode for the MVTU units
mem_mode = "decoupled"

# Apply HLS transformations
model = ModelWrapper("outputs/simple_nn_streamlined.onnx")
model = model.transform(to_hls.InferBinaryMatrixVectorActivation())
model = model.transform(to_hls.InferQuantizedMatrixVectorActivation())
model = model.transform(to_hls.InferLabelSelectLayer())
model = model.transform(to_hls.InferThresholdingLayer())
model = model.transform(to_hls.InferConvInpGen())
model = model.transform(to_hls.InferStreamingMaxPool())
model = model.transform(absorb.AbsorbConsecutiveTransposes())
model = model.transform(InferDataLayouts())
parent_model = model.transform(CreateDataflowPartition())

# Save the model and inspect node types
parent_model.save("outputs/simple_nn_dataflow_model.onnx")


# Apply folding attributes
model = ModelWrapper("outputs/simple_nn_dataflow_model.onnx")
fc_layers = model.get_nodes_by_op_type("MatrixVectorActivation")
folding = [
    (16, 3, 128),
    (32, 32, 128),
    (16, 32, 128),
    (16, 32, 128),
    (4, 32, 81),
    (1, 32, 2),
    (1, 4, 2),
    (1, 8, 128),
    (5, 1, 3),
]

for fcl, (pe, simd, ififodepth) in zip(fc_layers, folding):
    fcl_inst = getCustomOp(fcl)
    fcl_inst.set_nodeattr("PE", pe)
    fcl_inst.set_nodeattr("SIMD", simd)
    fcl_inst.set_nodeattr("inFIFODepth", ififodepth)

# Use same SIMD values for the sliding window operators
swg_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator")
for i in range(len(swg_layers)):
    swg_inst = getCustomOp(swg_layers[i])
    simd = folding[i][1]
    swg_inst.set_nodeattr("SIMD", simd)

model = model.transform(GiveUniqueNodeNames())
model.save("outputs/simple_nn_folded.onnx")

