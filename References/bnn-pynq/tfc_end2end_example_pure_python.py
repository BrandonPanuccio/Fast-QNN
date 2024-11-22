'''Hardware'''


'''Specialize Layers'''

thresh_node = model.get_nodes_by_op_type("Thresholding")[0]
thresh_node_inst = getCustomOp(thresh_node)
thresh_node_inst.set_nodeattr("preferred_impl_style", "hls")

''''''

'''This edit is for adding ZU to the part map'''
from finn.util.basic import pynq_part_map
print(pynq_part_map.keys())

''''''

'''Specialize Layers for the specific fpga'''

from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
model = model.transform(SpecializeLayers(fpga_part))

model.save(build_dir+"/tfc_w1_a1_specialize_layers.onnx")
showInNetron(build_dir+"/tfc_w1_a1_specialize_layers.onnx")

''''''

'''Nodes'''

fc_layers = model.get_nodes_by_op_type("MVAU_hls")
# (PE, SIMD, in_fifo_depth, out_fifo_depth, ramstyle) for each layer
config = [
    (16, 49, [16], [64], "block"),
    (8, 8, [64], [64], "auto"),
    (8, 8, [64], [64], "auto"),
    (10, 8, [64], [10], "distributed"),
]
for fcl, (pe, simd, ififo, ofifo, ramstyle) in zip(fc_layers, config):
    fcl_inst = getCustomOp(fcl)
    fcl_inst.set_nodeattr("PE", pe)
    fcl_inst.set_nodeattr("SIMD", simd)
    fcl_inst.set_nodeattr("inFIFODepths", ififo)
    fcl_inst.set_nodeattr("outFIFODepths", ofifo)
    fcl_inst.set_nodeattr("ram_style", ramstyle)

# set parallelism for input quantizer to be same as first layer's SIMD
inp_qnt_node = model.get_nodes_by_op_type("Thresholding_hls")[0]
inp_qnt = getCustomOp(inp_qnt_node)
inp_qnt.set_nodeattr("PE", 49)

''''''



'''This is the Builddddd'''

from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
model = ModelWrapper(build_dir+"/tfc_w1_a1_set_folding_factors.onnx")
model = model.transform(ZynqBuild(platform = pynq_board, period_ns = target_clk_ns))

''''''
Making the drivers
''''''

from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
model = model.transform(MakePYNQDriver("zynq-iodma"))

''''''

'''Pynq Deployment'''

from shutil import copy
from distutils.dir_util import copy_tree

# create directory for deployment files
deployment_dir = make_build_dir(prefix="pynq_deployment_")
model.set_metadata_prop("pynq_deployment_dir", deployment_dir)

# get and copy necessary files
# .bit and .hwh file
bitfile = model.get_metadata_prop("bitfile")
hwh_file = model.get_metadata_prop("hw_handoff")
deploy_files = [bitfile, hwh_file]

for dfile in deploy_files:
    if dfile is not None:
        copy(dfile, deployment_dir)

# driver.py and python libraries
pynq_driver_dir = model.get_metadata_prop("pynq_driver_dir")
copy_tree(pynq_driver_dir, deployment_dir)

''''''