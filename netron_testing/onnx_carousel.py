import os
from flask import Flask, render_template, redirect, url_for
import netron

app = Flask(__name__)

PROJ_NAME = "alexnet_1w1a"
# Directory containing ONNX files
ONNX_FOLDER = "/home/fastqnn/finn/notebooks/Fast-QNN/outputs/txaviour/"
ONNX_FOLDER = ONNX_FOLDER + "alexnet_1w1a_0/checkpoints"

# Get sorted ONNX files by creation date
def get_sorted_onnx_files(folder_path):
    onnx_files = [
        (f, os.path.getctime(os.path.join(folder_path, f)))
        for f in os.listdir(folder_path)
        if f.endswith('.onnx')
    ]
    sorted_files = sorted(onnx_files, key=lambda x: x[1])
    return [os.path.join(folder_path, f[0]) for f in sorted_files]

# Store sorted ONNX file paths
onnx_files = get_sorted_onnx_files(ONNX_FOLDER)
model_names = [os.path.basename(path).replace(PROJ_NAME+'_', '') for path in onnx_files]  # Extract model names
# Track used ports to open multiple instances
netron_ports = {}

# Route to render the main UI
@app.route('/')
def index():
    # Render the main UI with the model list
    return render_template('carousel.html', onnx_files=onnx_files, enumerate=enumerate, model_names=model_names, project_name=PROJ_NAME)

# Route to view the ONNX model at a specific index and start Netron instance
@app.route('/view/<int:index>')
def view_model(index):
    if index < 0 or index >= len(onnx_files):
        return "Index out of range", 404
    model_path = onnx_files[index]
    model_name = os.path.basename(model_path)

    # Assign a unique port for each model instance
    port = 20123 + index  # Starting port number, unique per model index
    netron_ports[model_name] = port

    # Start a Netron instance on the assigned port
    netron.start(model_path, address=("127.0.0.1",port), browse=False)

    return "Model loaded", 200

if __name__ == "__main__":
    app.run(debug=True, port=30129)
