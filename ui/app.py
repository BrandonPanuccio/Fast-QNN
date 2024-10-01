from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'secret_key'


# Routes for other pages (About, GitHub, Setup, Jupyter)
@app.route('/')
def about():
    return render_template('about.html')


@app.route('/code/github')
def code_github():
    return render_template('code_github.html')


@app.route('/code/setup')
def code_setup():
    return render_template('code_setup.html')


@app.route('/code/jupyter')
def code_jupyter():
    return render_template('code_jupyter.html')


@app.route('/system/how-to-use')
def system_how_to_use():
    return render_template('system_how_to_use.html')


# Multi-step form routes
@app.route('/system/launch', methods=['GET', 'POST'])
def step1():
    if request.method == 'POST':
        target_board = request.form.get('target_board')
        model_type = request.form.get('model_type')
        verify_accuracy = request.form.get('verify_accuracy')

        # Store the form data in session
        session['target_board'] = target_board
        session['model_type'] = model_type
        session['verify_accuracy'] = verify_accuracy

        # Handle unsupported boards
        if target_board == "other":
            return render_template('unsupported_board.html')

        # File uploads based on user choices
        if 'pth_file' in request.files:
            pth_file = request.files['pth_file']
            session['pth_file'] = pth_file.filename

        if 'model_file' in request.files:
            model_file = request.files['model_file']
            session['model_file'] = model_file.filename

        session['dataset'] = request.form.get('dataset')

        if session['model_type'] == 'pretrained' and session['verify_accuracy'] == 'yes' and session[
            'dataset'] == 'other':
            return render_template('unsupported_dataset.html')

        return redirect(url_for('step2'))

    return render_template('step1.html')


@app.route('/step2', methods=['GET', 'POST'])
def step2():
    if request.method == 'POST':
        return redirect(url_for('step3'))
    return render_template('step2.html')


@app.route('/step3', methods=['POST'])
def step3():
    return render_template('step3.html')


@app.route('/step4', methods=['POST'])
def step4():
    return render_template('step4.html')


if __name__ == '__main__':
    app.run(debug=True)
