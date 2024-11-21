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

let savedData = {};

function saveInputs() {
    if (!validateInputs()) {
        return;
    }

    const prjName = document.getElementById('prj_name').value.trim();
    const projectFolder = document.getElementById('project_folder').value.trim();
    const modelType = document.getElementById('model_type').value;
    const modelPyFile = document.getElementById('model_py_file').value.trim();
    const modelPthFile = document.getElementById('model_pth_file').value.trim();
    const torchVisionModel = document.getElementById('torch_vision_model').value.trim();
    const datasetType = document.getElementById('dataset_type').value;
    const customDataset = document.getElementById('custom_dataset').value.trim();
    const torchVisionDataset = document.getElementById('torch_vision_dataset').value.trim();

    savedData = {
        prjName,
        projectFolder,
        modelType,
        modelPyFile,
        modelPthFile,
        torchVisionModel,
        datasetType,
        customDataset,
        torchVisionDataset
    };

    alert("Inputs have been successfully saved.");
}

function getSummary() {
    if (Object.keys(savedData).length === 0) {
        alert("No data has been saved yet.");
        return;
    }

    let summaryHtml = `
        <ul>
            <li><strong>Project Name:</strong> ${savedData.prjName || "Not specified"}</li>
            <li><strong>Project Folder:</strong> ${savedData.projectFolder || "Not specified"}</li>
            <li><strong>Model Type:</strong> ${savedData.modelType || "Not specified"}</li>
            <li><strong>Model Python File:</strong> ${savedData.modelPyFile || "Not specified"}</li>
            <li><strong>Model PTH File:</strong> ${savedData.modelPthFile || "Not specified"}</li>
            <li><strong>TorchVision Model:</strong> ${savedData.torchVisionModel || "Not specified"}</li>
            <li><strong>Dataset Type:</strong> ${savedData.datasetType || "Not specified"}</li>
            <li><strong>Custom Dataset:</strong> ${savedData.customDataset || "Not specified"}</li>
            <li><strong>TorchVision Dataset:</strong> ${savedData.torchVisionDataset || "Not specified"}</li>
        </ul>
    `;

    document.getElementById('summaryResult').innerHTML = summaryHtml;
}

document.getElementById('projectSaveButton').addEventListener('click', saveInputs);
document.getElementById('modelSaveButton').addEventListener('click', saveInputs);
document.getElementById('datasetSaveButton').addEventListener('click', saveInputs);
document.getElementById('getSummaryButton').addEventListener('click', getSummary);



