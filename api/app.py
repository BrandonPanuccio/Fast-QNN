from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

@app.route('/')
def api_test():
    # Ensure this matches the HTML file name in the templates folder
    return render_template("ui_development.html")

@app.route('/run-command', methods=['POST'])
def run_command():
    """
    Endpoint to execute a command on localhost and return its output.
    """
    # Parse JSON data from the request body
    data = request.get_json()

    if not data or 'command' not in data:
        # Return error if JSON data or command is missing
        return jsonify({"error": "No command provided in the JSON request."}), 400

    command = data['command']

    try:
        # Execute the command locally and capture the output
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)

        # Return command output or error message
        if result.returncode == 0:
            return jsonify({"output": result.stdout})
        else:
            return jsonify({"error": result.stderr}), 400

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Command execution timed out."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask server
    app.run(host="127.0.0.1", port=5000)

