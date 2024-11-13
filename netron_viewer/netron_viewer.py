import os
from flask import Flask, render_template, redirect, url_for, request, session, flash
import netron

app = Flask(__name__)
app.secret_key = 'eh78ZMarx3wJcsscsLbJlg=='  # Required for session handling

# Route for initial setup page
@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if request.method == 'POST':
        # Get data from form and store in session
        session['proj_name'] = request.form.get('proj_name', 'alexnet_1w1a')
        session['onnx_folder'] = request.form.get('onnx_folder',
                                                  '/home/fastqnn/finn/notebooks/Fast-QNN/outputs/txaviour/alexnet_1w1a_0/checkpoints')
        session['netron_start_port'] = int(request.form.get('netron_start_port', 20123))

        # Indicate setup completion
        session['setup_complete'] = True
        return redirect(url_for('index'))

    # Render setup form with default values
    return render_template('setup.html', proj_name=session.get('proj_name', 'alexnet_1w1a'),
                           onnx_folder=session.get('onnx_folder', '/home/fastqnn/finn/notebooks/Fast-QNN/outputs/txaviour/alexnet_1w1a_0/checkpoints'),
                           netron_start_port=session.get('netron_start_port', 20123))


# Route to reset configuration and go back to setup
@app.route('/reset')
def reset():
    # Clear session to require setup on next visit
    session.pop('setup_complete', None)
    session.pop('proj_name', None)
    session.pop('onnx_folder', None)
    session.pop('netron_start_port', None)
    return redirect(url_for('setup'))


# Get sorted ONNX files by creation date, with error handling
def get_sorted_onnx_files(folder_path):
    try:
        onnx_files = [
            (f, os.path.getctime(os.path.join(folder_path, f)))
            for f in os.listdir(folder_path)
            if f.endswith('.onnx')
        ]
        sorted_files = sorted(onnx_files, key=lambda x: x[1])
        return [os.path.join(folder_path, f[0]) for f in sorted_files]
    except FileNotFoundError:
        flash("Error: The specified ONNX folder was not found. Please check the path and try again.", "danger")
        return []
    except PermissionError:
        flash("Error: Permission denied when accessing the ONNX folder. Please check the folder permissions.", "danger")
        return []


@app.route('/')
def index():
    # Redirect to setup if not completed
    if 'setup_complete' not in session:
        return redirect(url_for('setup'))

    # Retrieve sorted ONNX files and model names
    onnx_folder = session.get('onnx_folder')
    proj_name = session.get('proj_name')
    netron_start_port = session.get('netron_start_port')
    onnx_files = get_sorted_onnx_files(onnx_folder)
    model_names = [os.path.basename(path).replace(proj_name + '_', '') for path in onnx_files] if onnx_files else []

    return render_template('viewer.html', onnx_files=onnx_files, enumerate=enumerate, model_names=model_names,
                           project_name=proj_name, netron_start_port=netron_start_port)


# Route to view the ONNX model at a specific index and start Netron instance
@app.route('/view/<int:index>')
def view_model(index):
    # Redirect to setup if not completed
    if 'setup_complete' not in session:
        return redirect(url_for('setup'))

    onnx_folder = session.get('onnx_folder')
    netron_start_port = session.get('netron_start_port')
    proj_name = session.get('proj_name')

    onnx_files = get_sorted_onnx_files(onnx_folder)
    if index < 0 or index >= len(onnx_files):
        return "Index out of range", 404

    model_path = onnx_files[index]
    port = netron_start_port + index
    netron.start(model_path, address=("127.0.0.1", port), browse=False)

    return "Model loaded", 200


if __name__ == "__main__":
    app.run(debug=True, port=30129)
