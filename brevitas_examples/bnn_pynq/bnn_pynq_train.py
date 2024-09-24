# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import sys

import brevitas.onnx as bo
import torch
from finn.util.pytorch import ToTensor
from qonnx.core.datatype import DataType
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.core.modelwrapper import ModelWrapper

from brevitas_examples.bnn_pynq.trainer import Trainer

# Pytorch precision
torch.set_printoptions(precision=10)


# Util method to add mutually exclusive boolean
def add_bool_arg(parser, name, default):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no_" + name, dest=name, action="store_false")
    parser.set_defaults(**{name: default})


# Util method to pass None as a string and be recognized as None value
def none_or_str(value):
    if value == "None":
        return None
    return value


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def parse_args(args):
    parser = argparse.ArgumentParser(description="PyTorch MNIST/CIFAR10 Training")
    # I/O
    parser.add_argument("--datadir", default="./data/", help="Dataset location")
    parser.add_argument("--experiments", default="./experiments", help="Path to experiments folder")
    parser.add_argument("--dry_run", action="store_true", help="Disable output files generation")
    parser.add_argument("--log_freq", type=int, default=10)
    # Execution modes
    parser.add_argument(
        "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
    parser.add_argument(
        "--resume",
        dest="resume",
        type=none_or_str,
        help="Resume from checkpoint. Overrides --pretrained flag.")
    add_bool_arg(parser, "detect_nan", default=False)
    # Compute resources
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
    parser.add_argument("--gpus", type=none_or_str, default=None, help="Comma separated GPUs")
    # Optimizer hyperparams
    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    parser.add_argument("--lr", default=0.02, type=float, help="Learning rate")
    parser.add_argument("--optim", type=none_or_str, default="ADAM", help="Optimizer to use")
    parser.add_argument("--loss", type=none_or_str, default="SqrHinge", help="Loss function to use")
    parser.add_argument("--scheduler", default="FIXED", type=none_or_str, help="LR Scheduler")
    parser.add_argument(
        "--milestones", type=none_or_str, default='100,150,200,250', help="Scheduler milestones")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs")
    parser.add_argument("--random_seed", default=1, type=int, help="Random seed")
    # Neural network Architecture
    parser.add_argument("--network", default="LFC_1W1A", type=str, help="neural network")
    parser.add_argument("--pretrained", action='store_true', help="Load pretrained model")
    parser.add_argument("--strict", action='store_true', help="Strict state dictionary loading")
    parser.add_argument(
        "--state_dict_to_pth",
        action='store_true',
        help="Saves a model state_dict into a pth and then exits")
    return parser.parse_args(args)


class objdict(dict):

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def launch(cmd_args):
    args = parse_args(cmd_args)

    # Set relative paths relative to current workdir
    path_args = ["datadir", "experiments", "resume"]
    for path_arg in path_args:
        path = getattr(args, path_arg)
        if path is not None and not os.path.isabs(path):
            abs_path = os.path.abspath(os.path.join(os.getcwd(), path))
            setattr(args, path_arg, abs_path)

    # Access config as an object
    args = objdict(args.__dict__)

    # Avoid creating new folders etc.
    if args.evaluate:
        args.dry_run = True

    # Init trainer
    trainer = Trainer(args)

    # Execute
    if args.evaluate:
        with torch.no_grad():
            trainer.eval_model()
    else:
        trainer.train_model()

    torch.save(trainer.model.state_dict(), 'outputs/'+args.network+'.pth')

    # Export the model using brevitas
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor with MNIST dimensions
    bo.export_qonnx(trainer.model, input_tensor, 'outputs/'+args.network+'_export.onnx')

    # Load and transform the ONNX model
    model = ModelWrapper("outputs/"+args.network+"_export.onnx")
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    model.save("outputs/"+args.network+"_tidy.onnx")

    print('Model successfully exported and transformed')


    # Load and prepare the core model
    model = ModelWrapper("outputs/"+args.network+"_tidy.onnx")

    # Get input shape
    global_inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(global_inp_name)

    # Ensure input shape is in the correct format for ToTensor
    if isinstance(ishape, list):
        ishape = tuple(ishape)  # Convert list to tuple if needed

    # Prepare the ToTensor transformation
    totensor_pyt = ToTensor()

    # Export preprocessing model to ONNX
    chkpt_preproc_name = "outputs/"+args.network+"_preproc.onnx"
    try:
        bo.export_qonnx(totensor_pyt, torch.zeros(ishape), chkpt_preproc_name)  # Use a tensor with the shape
    except Exception as e:
        print(f"Error exporting ToTensor model: {e}")
        raise

    # Join preprocessing and core model
    pre_model = ModelWrapper(chkpt_preproc_name)
    model = model.transform(MergeONNXModels(pre_model))

    # Add input quantization annotation
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])

    # Save the final model
    model.save("outputs/"+args.network+"_with_preproc.onnx")

    print('Model with preprocessing successfully exported and transformed')

    # postprocessing: insert Top-1 node at the end
    model = model.transform(InsertTopK(k=1))
    chkpt_name = "outputs/"+args.network+"_pre_post.onnx"
    # tidy-up again
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    model.save(chkpt_name)

    from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
    from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
    from qonnx.core.modelwrapper import ModelWrapper

    test_pynq_board = "Pynq-Z2"
    target_clk_ns = 10

    from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild

    model = ModelWrapper(chkpt_name)
    # First, partition the model into streaming dataflow
    model = model.transform(CreateDataflowPartition())
    # Apply the ZynqBuild transformation
    model = model.transform(ZynqBuild(platform="Pynq-Z2", period_ns=10))

    # Apply the PYNQ Driver transformation
    model = model.transform(MakePYNQDriver("zynq-iodma"))

    # Save the final model
    model.save("outputs/"+args.network+"_synth.onnx")


def main():
    launch(sys.argv[1:])


if __name__ == "__main__":
    main()
