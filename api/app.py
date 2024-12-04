from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import inspect
from torchvision import datasets, models

app = Flask(__name__)
CORS(app)

# Output directories
OUTPUTS_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUTS_DIR, "models")
DATASET_DIR = os.path.join(OUTPUTS_DIR, "datasets")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# TorchVision data
torchvision_models = [
    model_name for model_name in dir(models)
    if not model_name.startswith("_") and callable(getattr(models, model_name))
]
available_datasets = [
    name for name, obj in inspect.getmembers(datasets, inspect.isclass)
    if issubclass(obj, datasets.VisionDataset)
]

# Progress state
progress_step = 0

def validate_file(file, allowed_extensions):
    """
    Validate the uploaded file for presence and allowed extensions.
    """
    if not file:
        return False, "No file provided."
    filename = file.filename
    if "." not in filename or filename.rsplit(".", 1)[1].lower() not in allowed_extensions:
        return False, f"Invalid file type. Allowed extensions are: {', '.join(allowed_extensions)}."
    return True, "File is valid."

@app.route('/setupproject', methods=['POST'])
def setup_project():
    """
    Handle the submission of project setup data and files.
    """
    global progress_step
    progress_step = 0  # Reset progress

    try:
        prj_name = request.form.get("prj_name")
        model_type = request.form.get("model_type")
        dataset_type = request.form.get("dataset_type")
        torch_vision_model = request.form.get("torch_vision_model", "").strip()
        torch_vision_dataset = request.form.get("torch_vision_dataset", "").strip()

        # Debugging logs
        print(f"[DEBUG] Received prj_name: {prj_name}, model_type: {model_type}, dataset_type: {dataset_type}")
        print(f"[DEBUG] Received torch_vision_model: {torch_vision_model}")
        print(f"[DEBUG] Received torch_vision_dataset: {torch_vision_dataset}")

        # Validate required fields
        if not prj_name or not model_type or not dataset_type:
            return jsonify({"error": "Missing required fields."}), 400

        # Validate TorchVision model
        if model_type == "torch_vision_pretrained":
            normalized_models = [model.lower() for model in torchvision_models]
            print(f"[DEBUG] Normalized models: {normalized_models}")
            if torch_vision_model.lower() not in normalized_models:
                return jsonify({
                    "error": f"Invalid TorchVision model: {torch_vision_model}. "
                             f"Available models: {', '.join(torchvision_models)}"
                }), 400
            torch_vision_model = torchvision_models[normalized_models.index(torch_vision_model.lower())]

        # Validate TorchVision dataset
        if dataset_type == "torch_vision_dataset":
            normalized_datasets = [dataset.lower() for dataset in available_datasets]
            print(f"[DEBUG] Normalized datasets: {normalized_datasets}")
            if torch_vision_dataset.lower() not in normalized_datasets:
                return jsonify({
                    "error": f"Invalid TorchVision dataset: {torch_vision_dataset}. "
                             f"Available datasets: {', '.join(available_datasets)}"
                }), 400
            torch_vision_dataset = available_datasets[normalized_datasets.index(torch_vision_dataset.lower())]

        # Validate and save files
        if model_type in ["untrained", "custom_pretrained"]:
            model_py_file = request.files.get("model_py_file")
            valid, message = validate_file(model_py_file, ["py"])
            if not valid:
                return jsonify({"error": message}), 400
            model_py_file.save(os.path.join(MODEL_DIR, f"{prj_name}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"))

        if model_type == "custom_pretrained":
            model_pth_file = request.files.get("model_pth_file")
            valid, message = validate_file(model_pth_file, ["pth"])
            if not valid:
                return jsonify({"error": message}), 400
            model_pth_file.save(os.path.join(MODEL_DIR, f"{prj_name}_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"))

        if dataset_type == "custom_dataset":
            custom_dataset = request.files.get("custom_dataset")
            valid, message = validate_file(custom_dataset, ["zip", "tar"])
            if not valid:
                return jsonify({"error": message}), 400
            custom_dataset.save(os.path.join(DATASET_DIR, f"{prj_name}_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"))

        return jsonify({"message": "Configuration received successfully!"}), 200
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/getupdate', methods=['GET'])
def get_update():
    """
    Provide incremental updates for the progress of the setup process.
    """
    global progress_step
    try:
        if progress_step < 10:
            progress_step += 1
            debug_message = f"Step {progress_step} completed successfully."
            print(f"[INFO] {debug_message}")
            return jsonify({"step": progress_step, "message": debug_message, "log": f"[DEBUG] {debug_message}"}), 200

        final_message = "All steps completed. Setup process finalized."
        print(f"[INFO] {final_message}")
        return jsonify({"step": progress_step, "message": final_message, "log": f"[DEBUG] {final_message}"}), 200
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(f"[ERROR] {error_message}")
        return jsonify({"error": error_message}), 500

@app.route('/autocomplete/models', methods=['GET'])
def autocomplete_models():
    """
    Return a list of TorchVision models for autocomplete functionality.
    """
    return jsonify({"models": torchvision_models}), 200

@app.route('/autocomplete/datasets', methods=['GET'])
def autocomplete_datasets():
    """
    Return a list of TorchVision datasets for autocomplete functionality.
    """
    return jsonify({"datasets": available_datasets}), 200

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)