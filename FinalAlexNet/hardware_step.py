from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.make_zynq_proj import MakeZYNQProject


test_pynq_board = "Pynq-Z2"
target_clk_ns = 10
model = ModelWrapper("outputs/alexnet_1w1a_mnist_folded.onnx")

# 1. Generate PYNQ driver
model = model.transform(MakePYNQDriver("outputs/alexnet_1w1a_mnist_pynq_z2_driver"))

# 2. Insert DMA and DWC nodes
model = model.transform(InsertIODMA())
model = model.transform(InsertDWC())

# 3. Partition for floorplanning
model = model.transform(Floorplan())
model = model.transform(CreateDataflowPartition())

# 4. Insert FIFO nodes and generate IP blocks
model = model.transform(InsertFIFO())
model = model.transform(PrepareIP())
model = model.transform(HLSSynthIP())

# 5. Create Vivado project and synthesize bitfile
model = model.transform(MakeZYNQProject())

# Save the final model ready for synthesis
model.save('outputs/alexnet_1w1a_mnist_final.onnx')

