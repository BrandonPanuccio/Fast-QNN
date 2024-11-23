// Toggles visibility of fields in the Model Configuration section
function toggleModelFields() {
    const modelType = document.getElementById("model_type").value;
    document.getElementById("model_py_file_group").style.display =
        (modelType === "untrained" || modelType === "custom_pretrained") ? "block" : "none";
    document.getElementById("model_pth_file_group").style.display =
        (modelType === "custom_pretrained") ? "block" : "none";
    document.getElementById("torch_vision_model_group").style.display =
        (modelType === "torch_vision_pretrained") ? "block" : "none";
}

function toggleDatasetFields() {
    const datasetType = document.getElementById("dataset_type").value;
    document.getElementById("custom_dataset_group").style.display =
        (datasetType === "custom_dataset") ? "block" : "none";
    document.getElementById("torch_vision_dataset_group").style.display =
        (datasetType === "torch_vision_dataset") ? "block" : "none";
}

function validateFileExtension(input, allowedExtensions) {
    const fileName = input.files[0] ? input.files[0].name : "";
    const fileExtension = fileName.split('.').pop().toLowerCase();
    return allowedExtensions.includes(fileExtension);
}

function clearErrors() {
    document.querySelectorAll(".error-message").forEach(el => el.textContent = "");
    document.querySelectorAll(".form-control").forEach(el => el.classList.remove("error-border"));
}

function setError(fieldId, message) {
    document.getElementById(fieldId).classList.add("error-border");
    document.getElementById(fieldId + "_error").textContent = message;
}

function validateInputs() {
    clearErrors();
    let isValid = true;

    const prjName = document.getElementById("prj_name").value;
    if (!prjName) {
        setError("prj_name", "Project name is required.");
        isValid = false;
    }

    const projectFolder = document.getElementById("project_folder").files;
    if (!projectFolder.length) {
        setError("project_folder", "Project folder selection is required.");
        isValid = false;
    }

    const modelType = document.getElementById("model_type").value;
    if (!modelType) {
        setError("model_type", "Model type is required.");
        isValid = false;
    }

    const modelPyFile = document.getElementById("model_py_file");
    if ((modelType === "untrained" || modelType === "custom_pretrained") && !modelPyFile.files.length) {
        setError("model_py_file", "Model Python file is required.");
        isValid = false;
    } else if (modelPyFile.files.length && !validateFileExtension(modelPyFile, ["py"])) {
        setError("model_py_file", "Model Python file must have a .py extension.");
        isValid = false;
    }

    const modelPthFile = document.getElementById("model_pth_file");
    if (modelType === "custom_pretrained" && !modelPthFile.files.length) {
        setError("model_pth_file", "Model weights file is required.");
        isValid = false;
    } else if (modelPthFile.files.length && !validateFileExtension(modelPthFile, ["pth"])) {
        setError("model_pth_file", "Model weights file must have a .pth extension.");
        isValid = false;
    }

    if (modelType === "torch_vision_pretrained" && !document.getElementById("torch_vision_model").value) {
        setError("torch_vision_model", "TorchVision model name is required.");
        isValid = false;
    }

    const datasetType = document.getElementById("dataset_type").value;
    if (!datasetType) {
        setError("dataset_type", "Dataset type is required.");
        isValid = false;
    }

    const customDataset = document.getElementById("custom_dataset");
    if (datasetType === "custom_dataset" && !customDataset.files.length) {
        setError("custom_dataset", "Custom dataset file is required.");
        isValid = false;
    } else if (customDataset.files.length && !validateFileExtension(customDataset, ["zip", "tar"])) {
        setError("custom_dataset", "Custom dataset file must be a .zip or .tar archive.");
        isValid = false;
    }

    if (datasetType === "torch_vision_dataset" && !document.getElementById("torch_vision_dataset").value) {
        setError("torch_vision_dataset", "TorchVision dataset name is required.");
        isValid = false;
    }

    return isValid;
}

document.getElementById('submitConfigButton').addEventListener('click', async function () {
    if (!validateInputs()) return;

    const configData = {
        prjName: document.getElementById('prj_name').value.trim(),
        projectFolder: document.getElementById('project_folder').value.trim(),
        modelType: document.getElementById('model_type').value,
        modelPyFile: document.getElementById('model_py_file').value.trim(),
        modelPthFile: document.getElementById('model_pth_file').value.trim(),
        torchVisionModel: document.getElementById('torch_vision_model').value.trim(),
        datasetType: document.getElementById('dataset_type').value,
        customDataset: document.getElementById('custom_dataset').value.trim(),
        torchVisionDataset: document.getElementById('torch_vision_dataset').value.trim()
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/submit-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(configData)
        });

        if (response.ok) {
            window.location.href = '/ui_step2.html';
        } else {
            alert('Failed to submit configuration.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while submitting the configuration.');
    }
});




