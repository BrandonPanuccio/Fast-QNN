import json
import shutil
import subprocess
import threading
import time
import webbrowser

import paramiko

from flask import Flask, send_from_directory, redirect, jsonify, request, Response
import os

from client.netron_viewer import netron_viewer

# Correcting the client and static folder paths explicitly based on Fast-QNN as the base
CLIENT_FOLDER = os.path.abspath(os.path.dirname(__file__))  # Path to client folder
TEMPLATES_FOLDER = os.path.join(CLIENT_FOLDER, "templates")
STATIC_FOLDER = os.path.join(CLIENT_FOLDER, "static")  # Path to the static folder
CSS_FOLDER = os.path.join(STATIC_FOLDER, "css") #Path to the css folder
JS_FOLDER = os.path.join(STATIC_FOLDER, "js") #Path to js folder

print(f"CLIENT_FOLDER: {CLIENT_FOLDER}")
print(f"STATIC_FOLDER: {TEMPLATES_FOLDER}")
print(f"STATIC_FOLDER: {STATIC_FOLDER}")
print(f"CSS_FOLDER: {CSS_FOLDER}")
print(f"JS_FOLDER: {JS_FOLDER}")

app = Flask(
    __name__,
    static_folder=STATIC_FOLDER,
    template_folder=TEMPLATES_FOLDER  # Serve HTML files from the client folder
)


@app.route("/")
def serve_root():
    """
    Redirect to the setup page.
    """
    return redirect("/setup")

@app.route("/setup")
def serve_setup():
    """
    Serve the setup.html file.
    """
    return send_from_directory(TEMPLATES_FOLDER, "setup.html")


@app.route("/steps")
def serve_steps():
    """
    Serve the steps.html file.
    """
    return send_from_directory(TEMPLATES_FOLDER, "steps.html")

@app.route("/deploy")
def serve_deploy():
    """
    Serve the steps.html file.
    """
    return send_from_directory(TEMPLATES_FOLDER, "deploy.html")

@app.route('/start_netron_viewer', methods=['GET'])
def start_netron_viewer():
    """
    Launch the Netron Viewer in a new tab.
    """
    try:
        proj_name = request.args.get('proj_name', 'alexnet_1w1a')
        zip_link = request.args.get('zip_link', None)

        # Construct the URL for Netron Viewer setup
        setup_url = f"http://127.0.0.1:30129/setup"
        if zip_link:
            setup_url += f"?proj_name={proj_name}&zip_link={zip_link}"

        # Run Netron Viewer in a separate thread
        def launch_netron_viewer():
            netron_viewer.run()

        threading.Thread(target=launch_netron_viewer, daemon=True).start()

        time.sleep(10)
        # Open the setup URL in a new browser tab
        webbrowser.open(setup_url)

        return jsonify({"message": "Netron Viewer started successfully.", "url": setup_url}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/list_usb_devices', methods=['GET'])
def list_usb_devices():
    try:
        # Use `lsblk` to get block devices and filter USB storage devices
        result = subprocess.run(['lsblk', '-o', 'NAME,MOUNTPOINT,TRAN', '--json'], capture_output=True, text=True)
        devices = []
        if result.returncode == 0:
            output = result.stdout
            data = json.loads(output)  # Parse the JSON-like output
            for block in data.get('blockdevices', []):
                if block.get('tran') == 'usb' and block.get('children', None) is not None:
                    for device in block.get('children', []):
                        if device.get('name', None) is not None and device.get('mountpoint', None) is not None:
                            devices.append({
                                'name': device.get('name'),
                                'mountpoint': device.get('mountpoint')
                            })
        return jsonify({'devices': devices}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/copy_to_usb', methods=['POST'])
def copy_to_usb():
    try:
        usb_device = request.form.get('usb_device')
        if not usb_device:
            return jsonify({"error": "USB device is required."}), 400

        zip_file = request.files.get('zip_file')
        if not zip_file:
            return jsonify({"error": "Zip file is required."}), 400

        # Ensure the USB device is mounted
        if not os.path.ismount(usb_device):
            return jsonify({"error": f"USB device {usb_device} is not mounted."}), 400

        # Define the destination path on the USB
        destination_path = os.path.join(usb_device, zip_file.filename)

        # Save the uploaded zip file to the USB device
        zip_file.save(destination_path)

        # Create a folder with the name of the zip file (excluding .zip)
        zip_folder_name = os.path.splitext(zip_file.filename)[0]  # Get name without .zip
        extraction_path = os.path.join(usb_device, zip_folder_name)
        os.makedirs(extraction_path, exist_ok=True)

        # Extract the zip file into the folder
        import zipfile
        with zipfile.ZipFile(destination_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)

        return jsonify({"message": f"File successfully copied and extracted to {extraction_path}."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def execute_ssh_commands_stream(board_ip, board_username, board_password, commands):
    """
    Stream SSH command output to the client.

    Parameters:
        board_ip (str): The IP address of the board.
        board_username (str): The username for SSH login.
        board_password (str): The password for SSH login.
        commands (list): A list of tuples where each tuple contains a command (str) and its timeout (int).

    Yields:
        str: Command output as it's generated.
    """
    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname=board_ip, username=board_username, password=board_password, timeout=10)

        shell = ssh_client.invoke_shell()
        time.sleep(1)

        yield "data: Connected to SSH successfully.<br>\n\n"

        for command, timeout in commands:
            yield f"data: Running: {command}<br>\n\n"
            shell.send(command + "\n")
            start_time = time.time()
            for _ in range(0, timeout, 2):
                if shell.recv_ready():
                    cmd_output = shell.recv(4096).decode().strip()
                    print(cmd_output)
                    yield f"data: {cmd_output}<br>\n\n"
                time.sleep(2)

            # Flush remaining output
            while shell.recv_ready():
                cmd_output = shell.recv(4096).decode().strip()
                print(cmd_output)
                yield f"data: {cmd_output}<br>\n\n"

        shell.close()
    except Exception as e:
        yield f"data: Error: {str(e)}<br>\n\n"
    finally:
        ssh_client.close()
        yield "data: END<br>\n\n"  # Explicit signal to indicate the end of the stream


@app.route('/deploy_to_board', methods=['GET'])
def deploy_to_board():
    try:
        board_ip = request.args.get("board_ip")
        board_username = request.args.get("board_username")
        board_password = request.args.get("board_password")

        if not board_ip or not board_username or not board_password:
            return jsonify({"error": "Missing required fields: board_ip, board_username, board_password"}), 400

        commands = [
            ("echo '" + board_password + "' | sudo -S su", 2),
            ("sudo -S su", 2),
            ("mkdir -p /media/usb", 2),
            ("USB_PART=$(lsblk -o NAME,TRAN | grep usb | awk '{print $1}' | head -n 1)", 2),
            ("mount /dev/${USB_PART}1 /media/usb", 2),
            ("cd /media/usb", 2),
            ("cd deploy_on_pynq", 2),
            ("source /etc/profile.d/pynq_venv.sh", 2),
            ("source /etc/profile.d/xrt_setup.sh", 2),
            ("rm -r ~/deploy_on_pynq/", 30),
            ("rsync -av ./ ~/deploy_on_pynq/", 30),
            ("cd ~/deploy_on_pynq/", 2),
            ("sudo umount /media/usb", 10),
            ("python3 validate.py --dataset_root './dataset' --dataset mnist --batchsize 1000 --bitfile ./resizer.bit", 90)
        ]

        return Response(
            execute_ssh_commands_stream(board_ip, board_username, board_password, commands),
            content_type="text/event-stream"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test_camera', methods=['GET'])
def test_camera():
    try:
        board_ip = request.args.get("board_ip")
        board_username = request.args.get("board_username")
        board_password = request.args.get("board_password")

        if not board_ip or not board_username or not board_password:
            return jsonify({"error": "Missing required fields: board_ip, board_username, board_password"}), 400

        commands = [
            ("echo '" + board_password + "' | sudo -S su", 2),
            ("sudo -S su", 2),
            ("source /etc/profile.d/pynq_venv.sh", 2),
            ("source /etc/profile.d/xrt_setup.sh", 2),
            ("cd ~/deploy_on_pynq/", 2),
            ("sudo kill -9 $(sudo lsof -t -i:5000)", 4),
            ("python3 camera.py --bitfile ./resizer.bit", 180)
        ]

        return Response(
            execute_ssh_commands_stream(board_ip, board_username, board_password, commands),
            content_type="text/event-stream"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5500)



