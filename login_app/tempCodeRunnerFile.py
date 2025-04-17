@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('login'))