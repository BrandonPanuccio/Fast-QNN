const API_BASE = "http://127.0.0.1:5000"; // API base URL

/**
 * Validate file input for allowed extensions.
 * @param {HTMLInputElement} fileInput - File input element.
 * @param {string[]} allowedExtensions - Array of allowed file extensions.
 * @returns {boolean} - True if valid, false otherwise.
 */
function validateFileType(fileInput, allowedExtensions) {
    if (!fileInput || !fileInput.files.length) return false;

    const fileName = fileInput.files[0].name;
    const fileExtension = fileName.split('.').pop().toLowerCase();
    return allowedExtensions.includes(fileExtension);
}

/**
 * Set an error message for a field.
 * @param {string} fieldId - The ID of the field to highlight.
 * @param {string} message - The error message to display.
 */
function setError(fieldId, message) {
    const errorElement = document.getElementById(`${fieldId}_error`);
    if (errorElement) {
        errorElement.textContent = message; // Set the error message
    }
    const field = document.getElementById(fieldId);
    if (field) {
        field.classList.add("error-border"); // Add red border to the field
    }
}

/**
 * Clear all error messages and remove red highlights from fields.
 */
function clearErrors() {
    document.querySelectorAll(".error-message").forEach((el) => (el.textContent = ""));
    document.querySelectorAll(".form-control").forEach((el) => el.classList.remove("error-border"));
}


/**
 * Populate a datalist element with options.
 * @param {string} datalistId - ID of the datalist element.
 * @param {string[]} options - List of options to populate.
 */
function populateDatalist(datalistId, options) {
    const datalist = document.getElementById(datalistId);
    datalist.innerHTML = ""; // Clear existing options
    options.forEach((option) => {
        const optionElement = document.createElement("option");
        optionElement.value = option;
        datalist.appendChild(optionElement);
    });
}

/**
 * Toggle file input fields based on model type selection.
 */
function toggleModelFields() {
    const modelType = document.getElementById("modelType").value;
    document.getElementById("model_py_file_group").style.display =
        modelType === "untrained" || modelType === "custom_pretrained" ? "block" : "none";
    document.getElementById("model_pth_file_group").style.display =
        modelType === "custom_pretrained" ? "block" : "none";
    document.getElementById("torch_vision_model_group").style.display =
        modelType === "torch_vision_pretrained" ? "block" : "none";
}

/**
 * Toggle dataset-specific fields based on dataset type selection.
 */
function toggleDatasetFields() {
    const datasetType = document.getElementById("datasetType").value;
    document.getElementById("custom_dataset_group").style.display =
        datasetType === "custom_dataset" ? "block" : "none";
    document.getElementById("torch_vision_dataset_group").style.display =
        datasetType === "torch_vision_dataset" ? "block" : "none";
}

/**
 * Fetch TorchVision models and datasets for autocomplete.
 */
async function fetchTorchVisionData() {
    try {
        // Fetch TorchVision models
        const modelsResponse = await fetch(`${API_BASE}/autocomplete/models`);
        if (modelsResponse.ok) {
            const modelsData = await modelsResponse.json();
            populateDatalist("torchvisionModelList", modelsData);
        }

        // Fetch TorchVision datasets
        const datasetsResponse = await fetch(`${API_BASE}/autocomplete/datasets`);
        if (datasetsResponse.ok) {
            const datasetsData = await datasetsResponse.json();
            populateDatalist("torchvisionDatasetList", datasetsData);
        }
    } catch (error) {
        console.error("Error fetching TorchVision data:", error);
    }
}

/**
 * Validate form inputs.
 * @returns {boolean} - True if valid, false otherwise.
 */
function validateInputs() {
    clearErrors();
    let isValid = true;

    // Validate project name
    const projectName = document.getElementById("projectName").value.trim();
    if (!projectName) {
        setError("projectName", "Project name is required.");
        isValid = false;
    }

    // Validate project folder
    const projectFolder = document.getElementById("project_folder").files;
    if (!projectFolder.length) {
        setError("project_folder", "Project folder is required.");
        isValid = false;
    }

    // Validate model type
    const modelType = document.getElementById("modelType").value;
    if (!modelType) {
        setError("modelType", "Model type is required.");
        isValid = false;
    }

    // Validate Python file if required
    const modelPyFile = document.getElementById("model_py_file");
    if ((modelType === "untrained" || modelType === "custom_pretrained") && !validateFileType(modelPyFile, ["py"])) {
        setError("model_py_file", "Python file (.py) is required.");
        isValid = false;
    }

    // Validate model weights file if required
    const modelPthFile = document.getElementById("model_pth_file");
    if (modelType === "custom_pretrained" && !validateFileType(modelPthFile, ["pth"])) {
        setError("model_pth_file", "Weights file (.pth) is required.");
        isValid = false;
    }

    // Validate TorchVision model name if required
    const torchVisionModel = document.getElementById("torch_vision_model").value.trim();
    if (modelType === "torch_vision_pretrained" && !torchVisionModel) {
        setError("torch_vision_model", "TorchVision model name is required.");
        isValid = false;
    }

    // Validate dataset type
    const datasetType = document.getElementById("datasetType").value;
    if (!datasetType) {
        setError("datasetType", "Dataset type is required.");
        isValid = false;
    }

    // Validate TorchVision dataset name if required
    const torchVisionDataset = document.getElementById("torch_vision_dataset").value.trim();
    if (datasetType === "torch_vision_dataset" && !torchVisionDataset) {
        setError("torch_vision_dataset", "TorchVision dataset name is required.");
        isValid = false;
    }

    // Validate custom dataset file if required
    const customDataset = document.getElementById("custom_dataset");
    if (datasetType === "custom_dataset" && !validateFileType(customDataset, ["zip", "tar"])) {
        setError("custom_dataset", "Custom dataset archive (.zip or .tar) is required.");
        isValid = false;
    }

    return isValid;
}

/**
 * Submit the configuration to the backend.
 */
async function submitConfiguration(event) {
    event.preventDefault(); // Prevent form refresh on submit

    if (!validateInputs()) return;

    const formData = new FormData();
    formData.append("prj_name", document.getElementById("projectName").value.trim());
    formData.append("model_type", document.getElementById("modelType").value);
    formData.append("dataset_type", document.getElementById("datasetType").value);

    const modelPyFile = document.getElementById("model_py_file").files[0];
    const modelPthFile = document.getElementById("model_pth_file").files[0];
    const customDataset = document.getElementById("custom_dataset").files[0];

    if (modelPyFile) formData.append("model_py_file", modelPyFile);
    if (modelPthFile) formData.append("model_pth_file", modelPthFile);
    if (customDataset) formData.append("custom_dataset", customDataset);

    // Append TorchVision fields
    const modelType = document.getElementById("modelType").value;
    const datasetType = document.getElementById("datasetType").value;

    if (modelType === "torch_vision_pretrained") {
        const torchVisionModel = document.getElementById("torch_vision_model").value.trim();
        formData.append("torch_vision_model", torchVisionModel);
    }
    if (datasetType === "torch_vision_dataset") {
        const torchVisionDataset = document.getElementById("torch_vision_dataset").value.trim();
        formData.append("torch_vision_dataset", torchVisionDataset);
    }

    try {
        const response = await fetch(`${API_BASE}/setupproject`, {
            method: "POST",
            body: formData,
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Unknown error occurred.");
        }

        alert(result.message);
        window.location.href = "steps.html";
    } catch (error) {
        console.error("Error submitting configuration:", error.message || error);
        alert(`Failed to submit configuration. Error: ${error.message || "Unknown error"}`);
    }
}

/**
 * Fetch updates for the steps.html page.
 */
async function fetchUpdates() {
    try {
        const response = await fetch(`${API_BASE}/getupdate`);
        if (!response.ok) throw new Error(`Failed to fetch updates: ${response.statusText}`);

        const result = await response.json();
        const updatesContainer = document.getElementById("progressUpdates");

        // Add new progress update on a new line
        updatesContainer.textContent += `\nStep ${result.step}: ${result.message}`;

        // Check if all steps are complete
        if (result.step < 10) {
            setTimeout(fetchUpdates, 5000);
        } else {
            updatesContainer.textContent += `\nAll done! Setup process is finalized.`;
        }
    } catch (error) {
        console.error("Error fetching updates:", error);
    }
}

// Attach event listeners
if (window.location.pathname.endsWith("setup.html")) {
    document.getElementById("submitButton").addEventListener("click", submitConfiguration);
    document.getElementById("modelType").addEventListener("change", toggleModelFields);
    document.getElementById("datasetType").addEventListener("change", toggleDatasetFields);
    fetchTorchVisionData();
} else if (window.location.pathname.endsWith("steps.html")) {
    fetchUpdates();
}


