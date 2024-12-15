import os
import time
from datetime import datetime

from pytz import timezone

from helper import *
from finn_flow import *

def finn_flow_task(prj_info):
    """
       Long-running task complete FINN flow.
       Logs progress updates to Redis for real-time updates.
       """
    try:
        log_message("Step 1: Initializing project setup")

        input_model = None
        input_model_shape = None

        if prj_info['Model_Type'] == "untrained":
            log_message("Training for untrained models are not supported at the moment!", level="error")
        elif prj_info['Model_Type'] == "custom_pretrained":
            log_message("Custom Pretrained models are not supported at the moment!", level="error")
        elif prj_info['Model_Type'] == "sample_pretrained" or prj_info['Model_Type'] == "finn_pretrained":
            pretrained_folder = os.path.join(prj_info['Folder'], "src")
            input_model, input_model_shape = load_pretrained_model(prj_info['Pretrained_Model'],
                                                                   prj_info['Model_Type'], pretrained_folder,
                                                                   initial_channels=1) #rgb =3, grayscale = 1 needs automation
        else:
            log_message("Unsupported Model Type", level="error")

        log_message("Step 2: Brevitas Export")

        export_onnx_path = set_onnx_checkpoint(prj_info, "Brevitas Export")
        export_qonnx(input_model, torch.randn(input_model_shape), export_onnx_path, opset_version=9)
        qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)

        log_message("Step 3: Convert QONNX to FINN")

        model = ModelWrapper(export_onnx_path)
        model = model.transform(ConvertQONNXtoFINN())
        model.save(set_onnx_checkpoint(prj_info, "Convert QONNX to FINN"))

        log_message("Step 4: Tidy ONNX Post Finn Conversion")
        model = tidy_up_transforms(model, set_onnx_checkpoint(prj_info, "Tidy ONNX Post Finn Conversion"))

        log_message("Step 5: Pre/Post Processing")
        set_onnx_checkpoint(prj_info, "Pre/Post Processing")
        global_inp_name = model.graph.input[0].name
        ishape = model.get_tensor_shape(global_inp_name)
        mul_model = MultiplyByOne()

        chkpt_mul_name = set_onnx_checkpoint(prj_info, "Mul_One_ONNX")
        export_qonnx(mul_model, torch.randn(ishape), chkpt_mul_name, opset_version=9)
        qonnx_cleanup(chkpt_mul_name, out_file=chkpt_mul_name)

        mul_model_wrapper = ModelWrapper(chkpt_mul_name)
        mul_model_wrapper = mul_model_wrapper.transform(ConvertQONNXtoFINN())
        model = model.transform(MergeONNXModels(mul_model_wrapper))

        global_inp_name = model.graph.input[0].name
        model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
        log_message("Skipping Pre-Processing. Will be expecting the user to handle it in application!", level="warning")

        model = model.transform(InsertTopK(k=1))
        model.save(set_onnx_checkpoint(prj_info, "Post Processing"))
        model = tidy_up_transforms(model, set_onnx_checkpoint(prj_info, "Tidy Post PrePost Proc"))

        log_message("Step 6: Streamlining ONNX")
        model = streamline_transforms(model, set_onnx_checkpoint(prj_info, "Streamlining ONNX"))

        log_message("Step 7: To HW Layers")
        model = to_hw_transforms(model, set_onnx_checkpoint(prj_info, "To HW Layers"))

        log_message("Step 8: Dataflow Partitioning")
        model = dataflow_partitioning(model, set_onnx_checkpoint(prj_info, "Dataflow Partition Parent Model"))
        model.save(set_onnx_checkpoint(prj_info, "Dataflow Partition Streaming Model"))

        log_message("Step 9: Specialize Model")
        model = specialize_layers_transform(model, prj_info['Board_name'], set_onnx_checkpoint(prj_info,
                                                                                                   f"Specialize Model Layers to {prj_info['Board_name']}"))

        log_message("Step 10: Folding Model")
        model = folding_transform(model, prj_info['Board_name'], set_onnx_checkpoint(prj_info, "Folded Model"))

        log_message("Step 11: Zynq Build (Will take anywhere between 30-120 minutes depending on modal size)")
        model = zynq_build_transform(model, set_onnx_checkpoint(prj_info, "Zynq Build"), prj_info['Board_name'])
        log_message("Step 12: Driver Creation")
        model = pynq_driver_transform(model, set_onnx_checkpoint(prj_info, "Pynq Driver"))


        log_message("Step 13: Preparing Deployment Files")
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
        make_archive(os.path.join(prj_info['Folder'], "output", "deploy_on_pynq"), 'zip', deployment_dir)
        # move zip to outputs directory
        '''
        log_message("Step 11: Zynq Build (Will take anywhere between 30-120 minutes depending on modal size)")
        time.sleep(15)
        log_message("Step 12: Driver Creation")
        log_message("Step 13: Preparing Deployment Files")
        '''
        make_archive(os.path.join(prj_info['Folder'], "output", "checkpoints"), 'zip', os.path.join(prj_info['Folder'], "checkpoints"))
        log_message("Project Setup Successfully")
        return "Done"

    except Exception as e:
        log_message(f"Error: {str(e)}", level="error")
        return "Error"

if __name__ == '__main__':
    prj_name_input = "Alexnet 3w3a MNIST"
    board_name_input = "Pynq-Z2"
    prj_folder_input = sanitize_string(prj_name_input)
    model_type_input = "sample_pretrained"
    pretrained_model_name_input = "alexnet_3w3a_mnist"
    Project_Info = setup_project(prj_name=prj_name_input, brd_name=board_name_input, model_type=model_type_input,
                                 sample_pretrained_model=pretrained_model_name_input)
    finn_flow_task(Project_Info)