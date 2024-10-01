from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.make_zynq_proj import MakeZYNQProject

# Load the model
model = ModelWrapper("outputs/alexnet_synth.onnx")

# Step 1: Insert IODMA and DWC nodes
model = model.transform(InsertIODMA())
model = model.transform(InsertDWC())

# Step 2: Partition the model for floorplanning
model = model.transform(CreateDataflowPartition())
model = model.transform(Floorplan())  # Optional: Helps with future floorplanning

# Step 3: Insert FIFOs between streaming nodes
model = model.transform(InsertFIFO())

# Step 4: Prepare the IP blocks (HLS layers and RTL layers)
model = model.transform(PrepareIP())   # Prepare IP generation for HLS and RTL
model = model.transform(HLSSynthIP())  # HLS synthesis for the HLS layers

# Step 5: Create the top-level stitched IP
model = model.transform(CreateStitchedIP())

# Step 6: Generate the Vivado project for Zynq
model = model.transform(MakeZYNQProject(platform="Pynq-Z2", period_ns=10))

# Step 7: Generate Python driver for PYNQ platform
model = model.transform(MakePYNQDriver("zynq-iodma"))

# Save the final model with transformations applied
model.save("outputs/alexnet_final.onnx")

print("Model successfully prepared for deployment on PYNQ.")
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.make_zynq_proj import MakeZYNQProject

# Load the model
model = ModelWrapper("outputs/alexnet_synth.onnx")

# Step 1: Insert IODMA and DWC nodes
model = model.transform(InsertIODMA())
model = model.transform(InsertDWC())

# Step 2: Partition the model for floorplanning
model = model.transform(CreateDataflowPartition())
model = model.transform(Floorplan())  # Optional: Helps with future floorplanning

# Step 3: Insert FIFOs between streaming nodes
model = model.transform(InsertFIFO())

# Step 4: Prepare the IP blocks (HLS layers and RTL layers)
model = model.transform(PrepareIP())   # Prepare IP generation for HLS and RTL
model = model.transform(HLSSynthIP())  # HLS synthesis for the HLS layers

# Step 5: Create the top-level stitched IP
model = model.transform(CreateStitchedIP())

# Step 6: Generate the Vivado project for Zynq
model = model.transform(MakeZYNQProject(platform="Pynq-Z2", period_ns=10))

# Step 7: Generate Python driver for PYNQ platform
model = model.transform(MakePYNQDriver("zynq-iodma"))

# Save the final model with transformations applied
model.save("outputs/alexnet_final.onnx")

print("Model successfully prepared for deployment on PYNQ.")
