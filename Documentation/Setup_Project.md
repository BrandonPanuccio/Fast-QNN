### Expected Inputs

1. `prj_name` (required, `str`)

   * **Description**: The name of the project.
   * **Purpose**: Used to generate a display name and a sanitized name (lowercase, no special characters) for storage.
   * **Condition**: Must be provided. Raises an error if not specified.

2. `project_folder` (required, `str`)

   * **Description**: The folder path where the project’s files will be stored.
   * **Purpose**: Ensures this directory exists and sets permissions to 777 if not already created.
   * **Condition**: Must be provided. Raises an error if not specified.

3. `model_type` (required, `str`)

   * **Description**: Specifies the type of model being used.
   * **Allowed Values**:
     * `"untrained"`: A new, untrained model defined in a custom Python file.
     * `"custom_pretrained"`: A pre-trained model that is custom-defined (not from TorchVision).
     * `"torch_vision_pretrained"`: A pre-trained model available from TorchVision.
   * **Condition**: Must be provided. Raises an error if not one of the allowed values.

4. `model_py_file` (conditional, `str`)

   * **Description**: The file name of the Python script that defines the model architecture.
   * **Purpose**: Used to locate and verify the existence of a model script for custom models.
   * **Condition**:
     * Required if `model_type` is `"untrained"` or `"custom_pretrained"`.
     * Raises a `FileNotFoundError` if not provided or does not exist in `project_folder`.

5. `model_pth_file` (conditional, `str`)

   * **Description**: The file name of the `.pth` file containing the pre-trained model weights.
   * **Purpose**: Required to load pre-trained weights for a custom-defined model.
   * **Condition**:
     * Required if `model_type` is `"custom_pretrained"`.
     * Raises a `FileNotFoundError` if not provided or does not exist in `project_folder`.

6. `torch_vision_model` (conditional, `str`)

   * **Description**: The name of a pre-trained model available in TorchVision.
   * **Purpose**: Specifies the pre-trained model architecture to be used.
   * **Condition**:
     * Required if `model_type` is `"torch_vision_pretrained"`.
     * Validated against the list of available TorchVision model names. Raises a `ValueError` if the specified model name is not available in TorchVision.

7. `dataset_type` (conditional, `str`)

   * **Description**: Specifies the type of dataset used for training.
   * **Allowed Values**:
     * `"torch_vision_dataset"`: A dataset provided by TorchVision.
     * `"custom_dataset"`: A custom dataset provided as an archive file.
   * **Purpose**: Determines if a TorchVision dataset or a custom dataset will be used.
   * **Condition**:
     * Required if `model_type` is `"untrained"`.
     * Raises a `ValueError` if not one of the allowed values.

8. `custom_dataset` (conditional, `str`)

   * **Description**: The file name of an archive (e.g., zip, tar) containing the custom dataset.
   * **Purpose**: Specifies the dataset file location for custom datasets.
   * **Condition**:
     * Required if `model_type` is `"untrained"` and `dataset_type` is `"custom_dataset"`.
     * Raises a `FileNotFoundError` if not provided, does not exist in `project_folder`, or is not a valid archive (zip or tar file).

9. `torch_vision_dataset` (conditional, `str`)

   * **Description**: The name of a TorchVision dataset class.
   * **Purpose**: Specifies the dataset to be loaded directly from TorchVision.
   * **Condition**:
     * Required if `model_type` is `"untrained"` and `dataset_type` is `"torch_vision_dataset"`.
     * Validated against the list of available TorchVision dataset names. Raises a `ValueError` if the specified dataset is not recognized.

---

### Summary Table

| Parameter            | Required | Condition                                                                                       |
|----------------------|----------|-------------------------------------------------------------------------------------------------|
| `prj_name`           | Yes      | Must be provided.                                                                              |
| `project_folder`     | Yes      | Must be provided.                                                                              |
| `model_type`         | Yes      | Must be one of `"untrained"`, `"custom_pretrained"`, `"torch_vision_pretrained"`.              |
| `model_py_file`      | Yes      | If `model_type` is `"untrained"` or `"custom_pretrained"`.                                     |
| `model_pth_file`     | Yes      | If `model_type` is `"custom_pretrained"`.                                                      |
| `torch_vision_model` | Yes      | If `model_type` is `"torch_vision_pretrained"`; validated against TorchVision models.         |
| `dataset_type`       | Yes      | If `model_type` is `"untrained"`; must be `"torch_vision_dataset"` or `"custom_dataset"`.      |
| `custom_dataset`     | Yes      | If `model_type` is `"untrained"` and `dataset_type` is `"custom_dataset"`; must be an archive. |
| `torch_vision_dataset` | Yes   | If `model_type` is `"untrained"` and `dataset_type` is `"torch_vision_dataset"`; validated against TorchVision datasets. |

This input specification ensures the function checks for the necessary parameters based on `model_type` and `dataset_type` while enforcing file and format requirements.
