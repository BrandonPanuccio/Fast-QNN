import netron
from finn.util.visualization import showSrc, showInNetron
from finn.util.basic import make_build_dir
import os

build_dir = "Fast-QNN/outputs/txaviour/AlexNet"

import torch
import onnx
from finn.util.test import get_test_model_trained
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup

alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
export_onnx_path = build_dir+"/tfc_w1_a1.onnx"
export_qonnx(alexnet, torch.rand(1, 3, 224, 224), build_dir+"/tfc_w1_a1.onnx")
qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)