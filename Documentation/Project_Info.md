## `Project_Info` Array

### Summary of `Project_Info` Contents

1. `Display_Name`

   * **Type**: `str`
   * **Description**: Stores the original, user-provided project name. This is used for display purposes and remains in its original format (case and characters intact).

2. `Stripped_Name`

   * **Type**: `str`
   * **Description**: Stores a sanitized version of the project name. It’s converted to lowercase, with special characters removed and spaces replaced by underscores, making it suitable for use in file names or identifiers.

3. `Folder`

   * **Type**: `str`
   * **Description**: Contains the absolute path to the `project_folder`. The function ensures this directory exists and sets its permissions to 777 if it didn’t already exist.

4. `Model_Type`

   * **Type**: `str`
   * **Description**: Stores the type of model specified by the user, which can be `"untrained"`, `"custom_pretrained"`, or `"torch_vision_pretrained"`. This information helps identify which files and parameters are relevant to the model.

5. `Model_Py_File` (if applicable)

   * **Type**: `str`
   * **Description**: Stores the filename of the Python script defining the model’s architecture. It is included if the `model_type` is `"untrained"` or `"custom_pretrained"` and exists in the `project_folder`.

6. `Model_Pth_File` (if applicable)

   * **Type**: `str`
   * **Description**: Contains the filename of the `.pth` file with the pre-trained model weights. This is relevant if `model_type` is `"custom_pretrained"` and the file exists in the `project_folder`.

7. `Torch_Vision_Model` (if applicable)

   * **Type**: `str`
   * **Description**: Stores the name of the pre-trained model from TorchVision, if `model_type` is `"torch_vision_pretrained"`. This is validated against TorchVision’s available models to ensure compatibility.

8. `Dataset_Type` (if applicable)

   * **Type**: `str`
   * **Description**: Indicates the dataset type used for training, which can be either `"torch_vision_dataset"` or `"custom_dataset"`. It’s required when `model_type` is `"untrained"`.

9. `Custom_Dataset` (if applicable)

   * **Type**: `str`
   * **Description**: Contains the filename of an archive file (e.g., zip, tar) that holds the custom dataset. This entry appears in `Project_Info` if `model_type` is `"untrained"` and `dataset_type` is `"custom_dataset"`.

10. `Torch_Vision_Dataset` (if applicable)

    * **Type**: `str`
    * **Description**: Stores the name of the dataset class from TorchVision if `dataset_type` is `"torch_vision_dataset"` and `model_type` is `"untrained"`. This dataset name is validated against the available datasets in TorchVision.

---

### Example `Project_Info` Output

After running `setup_project`, `Project_Info` may look like this:

```python
Project_Info = [
    {"Display_Name": "My Project"},
    {"Stripped_Name": "my_project"},
    {"Folder": "/absolute/path/to/project_folder"},
    {"Model_Type": "untrained"},
    {"Model_Py_File": "my_model.py"},
    {"Dataset_Type": "custom_dataset"},
    {"Custom_Dataset": "dataset.zip"}
]
