from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Global variable to store configuration data
config_data = {}

@app.route('/')
def home():
    """
    Renders the first UI page (ui_step1.html) where the user enters the configuration data.
    """
    return render_template("ui_step1.html")

@app.route('/submit-config', methods=['POST'])
def submit_config():
    """
    Receives configuration data from ui_step1.html and stores it in a global variable.
    Responds with a success message.
    """
    global config_data
    config_data = request.json  # Store the received data
    return jsonify({"message": "Configuration received successfully."}), 200

@app.route('/ui_step2.html')
def results_page():
    """
    Renders the second UI page (ui_step2.html) where the configuration data is displayed.
    """
    return render_template("ui_step2.html")

@app.route('/get-config', methods=['GET'])
def get_config():
    """
    Sends the stored configuration data to ui_step2.html for display.
    """
    return jsonify(config_data)

if __name__ == '__main__':
    # Run the Flask server on localhost at port 5000
    app.run(host="127.0.0.1", port=5000)


