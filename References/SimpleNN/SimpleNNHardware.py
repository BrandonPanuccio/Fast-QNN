from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from onnx import AttributeProto, NodeProto
from qonnx.core.modelwrapper import ModelWrapper

test_pynq_board = "Pynq-Z2"
target_clk_ns = 10

from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild

model = ModelWrapper("outputs/simple_nn_folded.onnx")
# First, partition the model into streaming dataflow
model = model.transform(CreateDataflowPartition())
# Apply the ZynqBuild transformation
model = model.transform(ZynqBuild(platform="Pynq-Z2", period_ns=10))

# Apply the PYNQ Driver transformation
model = model.transform(MakePYNQDriver("zynq-iodma"))

# Save the final model
model.save("outputs/simple_nn_synth.onnx")


