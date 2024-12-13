// API base URL
const API_BASE = "http://127.0.0.1:8999"; // Update as needed

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
 * Populate select options dynamically
 * @param {string} selectId - The ID of the select element
 * @param {Array} options - Array of options to populate
 */
function populateSelect(selectId, options) {
    const select = document.getElementById(selectId);
    select.innerHTML = "<option value=''>-- Select Option --</option>"; // Clear existing options and add a default

    // Ensure options is an array before iterating
    if (Array.isArray(options)) {
        options.forEach(option => {
            const opt = document.createElement("option");
            opt.value = option;
            opt.textContent = option;
            select.appendChild(opt);
        });
    } else {
        console.error(`Expected an array for options, but received:`, options);
    }
}

/**
 * Fetch TorchVision models, datasets, and boards for autocomplete and dynamic population.
 */
async function fetchConfigurationData() {
    try {
        // Fetch sample pretrained models
        const samplePretrainedModelsResponse = await fetch(`${API_BASE}/autocomplete/sample_pretrained_models`);
        if (samplePretrainedModelsResponse.ok) {
            const samplePretrainedModelsData = await samplePretrainedModelsResponse.json();
            populateSelect("sample_pretrained_model", samplePretrainedModelsData.sample_pretrained_models);
        }

        // Fetch sample untrained models
        const sampleUntrainedModelsResponse = await fetch(`${API_BASE}/autocomplete/sample_untrained_models`);
        if (sampleUntrainedModelsResponse.ok) {
            const sampleUntrainedModelsData = await sampleUntrainedModelsResponse.json();
            populateSelect("sample_untrained_model", sampleUntrainedModelsData.sample_untrained_models);
        }

        // Fetch FINN pretrained models
        const finnPretrainedModelsResponse = await fetch(`${API_BASE}/autocomplete/finn_pretrained_models`);
        if (finnPretrainedModelsResponse.ok) {
            const finnPretrainedModelsData = await finnPretrainedModelsResponse.json();
            populateSelect("finn_pretrained_model", finnPretrainedModelsData.finn_pretrained_models);
        }

        // Fetch TorchVision datasets
        const datasetsResponse = await fetch(`${API_BASE}/autocomplete/datasets`);
        if (datasetsResponse.ok) {
            const datasetsData = await datasetsResponse.json();
            populateSelect("torch_vision_dataset", datasetsData.datasets);
        }

        // Fetch available boards
        const boardsResponse = await fetch(`${API_BASE}/autocomplete/boards`);
        if (boardsResponse.ok) {
            const boardsData = await boardsResponse.json();
            populateSelect("boardName", boardsData.boards);
        }

    } catch (error) {
        console.error("Error fetching configuration data:", error);
    }
}

function toggleModelFields() {
    const modelType = document.getElementById("modelType").value;

    // Show/hide model configuration inputs based on selected model type
    document.getElementById("model_py_file_group").style.display =
        modelType === "untrained" || modelType === "custom_pretrained" ? "block" : "none";
    document.getElementById("model_pth_file_group").style.display =
        modelType === "custom_pretrained" ? "block" : "none";
    document.getElementById("sample_pretrained_model_group").style.display =
        modelType === "sample_pretrained" ? "block" : "none";
    document.getElementById("sample_untrained_model_group").style.display =
        modelType === "sample_untrained" ? "block" : "none";
    document.getElementById("finn_pretrained_model_group").style.display =
        modelType === "finn_pretrained" ? "block" : "none";

    // Show/hide dataset configuration section
    document.getElementById("dataset_configuration_section").style.display =
        modelType === "untrained" || modelType === "sample_untrained" ? "block" : "none";
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

    // Validate board name
    const boardName = document.getElementById("boardName").value;
    if (!boardName) {
        setError("boardName", "Board name is required.");
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

    // Validate sample pretrained model if required
    const samplePretrainedModel = document.getElementById("sample_pretrained_model").value;
    if (modelType === "sample_pretrained" && !samplePretrainedModel) {
        setError("sample_pretrained_model", "Sample pretrained model is required.");
        isValid = false;
    }

    // Validate sample untrained model if required
    const sampleUntrainedModel = document.getElementById("sample_untrained_model").value;
    if (modelType === "sample_untrained" && !sampleUntrainedModel) {
        setError("sample_untrained_model", "Sample untrained model is required.");
        isValid = false;
    }

    // Validate FINN pretrained model if required
    const finnPretrainedModel = document.getElementById("finn_pretrained_model").value;
    if (modelType === "finn_pretrained" && !finnPretrainedModel) {
        setError("finn_pretrained_model", "FINN pretrained model is required.");
        isValid = false;
    }

    // Validate dataset type
    const datasetType = document.getElementById("datasetType").value;
    if (!datasetType && (modelType === "sample_untrained" || modelType === "untrained")) {
        setError("datasetType", "Dataset type is required.");
        isValid = false;
    }

    // Validate TorchVision dataset name if required
    const torchVisionDataset = document.getElementById("torch_vision_dataset").value.trim();
    if (datasetType === "torch_vision_dataset" && !torchVisionDataset && (modelType !== "sample_untrained" || modelType !== "untrained")) {
        setError("torch_vision_dataset", "TorchVision dataset name is required.");
        isValid = false;
    }

    // Validate custom dataset file if required
    const customDataset = document.getElementById("custom_dataset");
    if (datasetType === "custom_dataset" && !validateFileType(customDataset, ["zip", "tar"]) && (modelType === "sample_untrained" || modelType === "untrained")) {
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
    formData.append("brd_name", document.getElementById("boardName").value);
    formData.append("model_type", document.getElementById("modelType").value);
    formData.append("dataset_type", document.getElementById("datasetType").value);

    // Append relevant files if provided
    const modelPyFile = document.getElementById("model_py_file").files[0];
    if (modelPyFile) formData.append("model_py_file", modelPyFile);

    const modelPthFile = document.getElementById("model_pth_file").files[0];
    if (modelPthFile) formData.append("model_pth_file", modelPthFile);

    const customDataset = document.getElementById("custom_dataset").files[0];
    if (customDataset) formData.append("custom_dataset", customDataset);

    // Append additional inputs based on model and dataset types
    const modelType = document.getElementById("modelType").value;
    const datasetType = document.getElementById("datasetType").value;

    if (modelType === "sample_pretrained") {
        const samplePretrainedModel = document.getElementById("sample_pretrained_model").value;
        formData.append("sample_pretrained_model", samplePretrainedModel);
    }

    if (modelType === "sample_untrained") {
        const sampleUntrainedModel = document.getElementById("sample_untrained_model").value;
        formData.append("sample_untrained_model", sampleUntrainedModel);
    }

    if (modelType === "finn_pretrained") {
        const finnPretrainedModel = document.getElementById("finn_pretrained_model").value;
        formData.append("finn_pretrained_model", finnPretrainedModel);
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
        sessionStorage.setItem("project_info", result.project_info); // Store project info for view
        window.location.href = "/steps"; // Navigate to steps.html

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

        if(result.message)
            updatesContainer.innerHTML += `\n${result.message}`;  // Add new progress update on a new line

        // Check if all steps are complete
        if (result.status === "UNKNOWN") {
            updatesContainer.innerHTML += `\nProgress Unknown. Try Again!`;
        } else if (result.status === "IN_PROGRESS"){
            setTimeout(fetchUpdates, 2000);
        } else if (result.status === "ERROR"){
            updatesContainer.innerHTML += `\nWe encountered an error. Please try again!`;
        } else{
            updatesContainer.innerHTML += `\nAll done!\n<a href='${API_BASE}${result.download_link}' target='_blank'>Download Deployment Zip</a>`;
            updatesContainer.innerHTML += `\n<a href='${API_BASE}${result.checkpoint_link}' target='_blank'>Download Checkpoints Zip</a>`;
        }
    } catch (error) {
        console.error("Error fetching updates:", error);
    }
}


/**
 * Populate summary table in steps.html.
 */
/**
 * Populate summary table in steps.html.
 */
function fetchSummary() {
    const summaryTable = document.getElementById("summaryTable");
    if (!summaryTable) {
        console.error("Summary table not found on the page.");
        return;
    }

    // Retrieve matching field names from localStorage
    const projectName = localStorage.getItem("projectName");
    const modelType = localStorage.getItem("modelType");
    const datasetType = localStorage.getItem("datasetType");
    const model_py_file = localStorage.getItem("model_py_file");
    const model_pth_file = localStorage.getItem("model_pth_file");
    const torch_vision_model = localStorage.getItem("torch_vision_model");
    const custom_dataset = localStorage.getItem("custom_dataset");
    const torch_vision_dataset = localStorage.getItem("torch_vision_dataset");

    const rows = `
        <tr><td>Project Name</td><td>${projectName || "N/A"}</td></tr>
        <tr><td>Model Type</td><td>${modelType || "N/A"}</td></tr>
        <tr><td>Dataset Type</td><td>${datasetType || "N/A"}</td></tr>
        <tr><td>.py File</td><td>${model_py_file || "N/A"}</td></tr>
        <tr><td>.pth File</td><td>${model_pth_file|| "N/A"}</td></tr>
        <tr><td>Torchvision Model</td><td>${torch_vision_model || "N/A"}</td></tr>
        <tr><td>Custom Dataset</td><td>${custom_dataset || "N/A"}</td></tr>
        <tr><td>Torchvision Dataset</td><td>${torch_vision_dataset || "N/A"}</td></tr>
    `;
    summaryTable.innerHTML = rows;
}


/**
 * Exit button behavior.
 */
function handleExitButtonClick() {
    alert("Exiting setup process...");
    window.location.href = "/setup";
}

// Attach event listeners
if (window.location.pathname.endsWith("/setup")) {
    document.getElementById("submitButton").addEventListener("click", submitConfiguration);
    document.getElementById("modelType").addEventListener("change", toggleModelFields);
    document.getElementById("datasetType").addEventListener("change", toggleDatasetFields);
    fetchConfigurationData();
}
else if (window.location.pathname.endsWith("/steps")) {
    fetchSummary();
    fetchUpdates();
    document.getElementById("exitButton").addEventListener("click", handleExitButtonClick);
}




