from flask import Flask, send_from_directory, redirect
import os

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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5500)



