import os
import time
from datetime import datetime

from pytz import timezone

from helper import *
from finn_flow import *
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

# Initialize the scheduler
scheduler = BackgroundScheduler({'apscheduler.timezone': 'UTC'})
scheduler.start()

# Store task statuses and logs
task_status = "UNKNOWN"
download_link = ""
checkpoint_link = ""
app = Flask(__name__)
CORS(app)

@app.route('/setupproject', methods=['POST'])
def api_setup():
    """
    Handle the submission of project setup data and files.
    """
    global task_status
    try:
        # Extract form data
        prj_name = request.form.get("prj_name")
        brd_name = request.form.get("brd_name")
        model_type = request.form.get("model_type")
        dataset_type = request.form.get("dataset_type")
        sample_untrained_model = request.form.get("sample_untrained_model", "").strip()
        sample_pretrained_model = request.form.get("sample_pretrained_model", "").strip()
        finn_pretrained_model = request.form.get("finn_pretrained_model", "").strip()
        torch_vision_dataset = request.form.get("torch_vision_dataset", "").strip()

        # Debugging logs
        print(f"[DEBUG] Received prj_name: {prj_name}, brd_name: {brd_name}, model_type: {model_type}, dataset_type: {dataset_type}")
        print(f"[DEBUG] Received sample_untrained_model: {sample_untrained_model}")
        print(f"[DEBUG] Received sample_pretrained_model: {sample_pretrained_model}")
        print(f"[DEBUG] Received finn_pretrained_model: {finn_pretrained_model}")
        print(f"[DEBUG] Received torch_vision_dataset: {torch_vision_dataset}")

        # Validate required fields
        if not prj_name or not brd_name or not model_type:
            return jsonify({"error": "Missing required fields."}), 400

        # Validate Board Name
        if brd_name not in available_boards:
            return jsonify({"error": f"Invalid board name '{brd_name}'. Available board names are: {', '.join(available_boards)}"}), 400

        # Validate Model Type
        if model_type not in valid_model_types:
            return jsonify({"error": f"Invalid model type '{model_type}'. Model type must be one of {valid_model_types}"}), 400

        # Handle Model Type specific requirements and validate corresponding fields
        model_py_file_path = None
        model_pth_file_path = None
        custom_dataset_path = None

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # for production mode
        timestamp = "0"
        prj_name_stripped = sanitize_string(prj_name)
        project_folder = f"{WORKING_FOLDER}{prj_name_stripped}_{timestamp}"
        folder_path = ensure_directory_exists(project_folder)
        model_dir = os.path.join(folder_path, "src")
        dataset_dir = os.path.join(folder_path, "dataset")

        if model_type in ["untrained", "custom_pretrained"]:
            model_py_file = request.files.get("model_py_file")
            if not model_py_file:
                return jsonify({"error": "Model Python file is required for 'untrained' or 'custom_pretrained' model types."}), 400
            valid, message = validate_file(model_py_file, ["py"])
            if not valid:
                return jsonify({"error": message}), 400
            model_py_file_path = os.path.join(model_dir, f"{prj_name}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
            model_py_file.save(model_py_file_path)

        if model_type == "custom_pretrained":
            model_pth_file = request.files.get("model_pth_file")
            if not model_pth_file:
                return jsonify({"error": "Model PTH file is required for 'custom_pretrained' model type."}), 400
            valid, message = validate_file(model_pth_file, ["pth"])
            if not valid:
                return jsonify({"error": message}), 400
            model_pth_file_path = os.path.join(model_dir, f"{prj_name}_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            model_pth_file.save(model_pth_file_path)

        if model_type == "sample_pretrained":
            if not sample_pretrained_model or sample_pretrained_model not in sample_pretrained_models:
                return jsonify({"error": f"Invalid sample pretrained model '{sample_pretrained_model}'. Available models are: {', '.join(sample_pretrained_models)}"}), 400

        if model_type == "sample_untrained":
            if not sample_untrained_model or sample_untrained_model not in sample_untrained_models:
                return jsonify({"error": f"Invalid sample untrained model '{sample_untrained_model}'. Available models are: {', '.join(sample_untrained_models)}"}), 400

        if model_type == "finn_pretrained":
            if not finn_pretrained_model or finn_pretrained_model not in finn_pretrained_models:
                return jsonify({"error": f"Invalid FINN pretrained model '{finn_pretrained_model}'. Available models are: {', '.join(finn_pretrained_models)}"}), 400

        # Validate Dataset Type and handle dataset-related requirements
        if model_type in ["untrained", "sample_untrained"]:
            if dataset_type not in ["torch_vision_dataset", "custom_dataset"]:
                return jsonify({"error": "Invalid dataset type. Dataset type must be either 'torch_vision_dataset' or 'custom_dataset'."}), 400

            if dataset_type == "custom_dataset":
                custom_dataset = request.files.get("custom_dataset")
                if not custom_dataset:
                    return jsonify({"error": "Custom dataset file is required when dataset type is 'custom_dataset'."}), 400
                valid, message = validate_file(custom_dataset, ["zip", "tar"])
                if not valid:
                    return jsonify({"error": message}), 400
                custom_dataset_path = os.path.join(dataset_dir, f"{prj_name}_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
                custom_dataset.save(custom_dataset_path)

            elif dataset_type == "torch_vision_dataset":
                if not torch_vision_dataset or torch_vision_dataset.lower() not in available_datasets:
                    return jsonify({"error": f"Invalid Torch Vision dataset '{torch_vision_dataset}'. Available datasets are: {', '.join(available_datasets)}"}), 400
        try:
            # Call setup_project() with the gathered inputs
            prj_info = setup_project(
                prj_name=prj_name,
                brd_name=brd_name,
                model_type=model_type,
                project_folder=folder_path,  # Let setup_project create the default folder
                model_py_file=model_py_file_path if model_py_file_path else None,
                model_pth_file=model_pth_file_path if model_pth_file_path else None,
                sample_untrained_model=sample_untrained_model if model_type == "sample_untrained" else None,
                sample_pretrained_model=sample_pretrained_model if model_type == "sample_pretrained" else None,
                finn_pretrained_model=finn_pretrained_model if model_type == "finn_pretrained" else None,
                dataset_type=dataset_type,
                custom_dataset=custom_dataset_path if dataset_type == "custom_dataset" else None,
                torch_vision_dataset=torch_vision_dataset if dataset_type == "torch_vision_dataset" else None,
            )
            r_msg = get_logs()
            # Generate a unique task ID
            task_id = f"task-{int(time.time())}"
            task_status = "IN_PROGRESS"

            # Add the background task to the scheduler
            scheduler.add_job(
                func=finn_flow_task,
                args=[prj_info],
                id=task_id,
                replace_existing=True,
            )
        except ValueError as e:
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 400
        else:
            return jsonify({
                "message": r_msg,
                "project_info": prj_info
            }), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/getupdate', methods=['GET'])
def get_update():
    try:
        if task_status == "DONE":
            return jsonify({"status": task_status, "message": f"{get_logs()}", "download_link": download_link, "checkpoint_link": checkpoint_link}), 200
        return jsonify({"status": task_status, "message": f"{get_logs()}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download', methods=['GET'])
def download_file():
    """
    Endpoint to download a file from a specified file location.
    """
    try:
        # Get the file path from query parameters
        file_path = request.args.get('file_path')

        if not file_path:
            return jsonify({"error": "File path is required"}), 400

        # Check if the file exists
        if not os.path.isfile(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404

        # Send the file to the client
        return send_file(file_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def finn_flow_task(prj_info):
    """
       Long-running task complete FINN flow.
       Logs progress updates to Redis for real-time updates.
       """
    global task_status
    global download_link
    global checkpoint_link
    try:
        task_status = "IN_PROGRESS"
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
        model = folding_transform(model, set_onnx_checkpoint(prj_info, "Folded Model"))

        '''
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

        download_link = "/download?file_path="+os.path.join(prj_info['Folder'], "output", "deploy_on_pynq.zip")
        make_archive(os.path.join(prj_info['Folder'], "output", "checkpoints"), 'zip', os.path.join(prj_info['Folder'], "checkpoints"))
        checkpoint_link = "/download?file_path="+os.path.join(prj_info['Folder'], "output", "checkpoints.zip")
        log_message("Project Setup Successfully")
        task_status = "DONE"
        return {"status": "DONE", "message": f"Project set up successfully."}

    except Exception as e:
        task_status = "ERROR"
        log_message(f"Error: {str(e)}", level="error")
        return {"status": "ERROR", "message": f"There was an error encountered. Please try again!"}


@app.route('/autocomplete/boards', methods=['GET'])
def autocomplete_boards():
    """
    Return a list of available boards for autocomplete functionality.
    """
    return jsonify({"boards": available_boards}), 200

@app.route('/autocomplete/sample_pretrained_models', methods=['GET'])
def autocomplete_sample_pretrained_models():
    """
    Return a list of sample pretrained models for autocomplete functionality.
    """
    return jsonify({"sample_pretrained_models": sample_pretrained_models}), 200

@app.route('/autocomplete/sample_untrained_models', methods=['GET'])
def autocomplete_sample_untrained_models():
    """
    Return a list of sample untrained models for autocomplete functionality.
    """
    return jsonify({"sample_untrained_models": sample_untrained_models}), 200

@app.route('/autocomplete/finn_pretrained_models', methods=['GET'])
def autocomplete_finn_pretrained_models():
    """
    Return a list of FINN pretrained models for autocomplete functionality.
    """
    return jsonify({"finn_pretrained_models": finn_pretrained_models}), 200

@app.route('/autocomplete/datasets', methods=['GET'])
def autocomplete_datasets():
    """
    Return a list of TorchVision datasets for autocomplete functionality.
    """
    return jsonify({"datasets": available_datasets}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8999)