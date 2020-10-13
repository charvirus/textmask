from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from flask import request
from flask import make_response

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/images/<path:path>')
def send_images(path):
    return send_from_directory('images', path)

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received client id : ' + request.sid)
    socketio.emit('my response', json)

if __name__ == '__main__':
    socketio.run(app, debug=True)
    