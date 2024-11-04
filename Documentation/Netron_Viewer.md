# Netron Viewer Documentation

The Netron Viewer is a Flask-based web application that allows users to visually explore ONNX model files using Netron. This app provides an interactive user interface to load, view, compare, and organize ONNX models in a custom-defined directory.

---

## Table of Contents
- [Getting Started](#getting-started)
- [Setup Page](#setup-page)
- [Viewer Page](#viewer-page)
- [Key Features](#key-features)
- [Error Handling](#error-handling)

---

## Getting Started

To use the Netron Viewer, follow these steps:

1. Ensure you are not in the Finn Environment or the docker container.
2. Install the necessary dependencies:
   - **Flask, Netron, Jinja2**
     ```bash
     pip install flask netron jinja2
     ```
3. Run `netron_viewer.py` by executing:
   ```bash
   python netron_viewer.py

## Setup Page

The Setup Page (`setup.html`) allows you to define key configuration settings for the Netron Viewer:

1. **Project Name**: The unique identifier for the project, used to label models within the viewer.
2. **ONNX Folder Path**: The absolute path to the directory containing ONNX model files. You can:
   - Use the "Choose Folder" button to select a file within the target directory (only supported in WebKit browsers).
   - Manually enter the directory path if using non-WebKit browsers.
3. **Netron Starting Port**: The starting port number for Netron to serve models. Each model view will increment the port by one.

Once configured, click **Save and View Models** to load the Viewer Page.

---

## Viewer Page

The Viewer Page (`viewer.html`) displays all the ONNX models found in the specified directory in a grid layout. Each model card has three primary functions:

1. **Open**: Opens the selected model in a new Netron window.
2. **Close**: Closes the currently open Netron window for the model.
3. **Compare**: Opens multiple models side-by-side, with windows arranged to fit on the screen.

Additional global options:
- **Show All Open**: Arranges all currently open Netron windows side-by-side on the screen.
- **Close All**: Closes all currently open Netron windows.
- **Reset and Go to Setup**: Resets the configuration and redirects to the Setup Page.

---

## Key Features

1. **Dynamic Port Allocation**: Each ONNX model is loaded on a unique port, starting from the defined starting port.
2. **Window Arrangement**: The `Show All Open` and `Compare` buttons arrange open windows side-by-side for easy model comparison.
3. **Reset Functionality**: The `Reset and Go to Setup` button closes all open windows and redirects to the Setup Page to change configuration settings.

---

## Error Handling

- **Folder Not Found or Permission Denied**: If the specified folder path is invalid or inaccessible, an error message is displayed on the Setup Page.
- **Index Out of Range**: If a model index is invalid, a "404 Index out of range" error is returned.