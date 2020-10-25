from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/send')
def send():
    return render_template('post.html')

@app.route('/post', methods=['POST'])
def post():
    value = request.form['send']
    return value

if __name__ == '__main__':
    app.run()