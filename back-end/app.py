from flask import Flask, redirect, request, url_for, jsonify, session
from flask import app, Response, send_file
from flask_cors import CORS
from open_track import Open_track
from load_model import CNN
from werkzeug.utils import secure_filename
import os 

# <--- global varible area ---> 
UPLOAD_FOLDER = '../front-end/src'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# <--- start flask --->
app = Flask(__name__)
CORS(app)

# <--- app config area --->
app.config["CLIENT_WAV"] = "/home/micro/harmonica_train/harmonica_project/back-end/static/HarmonicaData/wav/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return 'HOME'

def Opentrack(wav_name):
        open = Open_track()
        tensor_track = open.readsound(wav_name)

        model = open.loadmodel()
        
        output = open.putincnn(tensor_track,model)

        Dout = open.findwrongnote(output) 
        return jsonify(Dout)


@app.route('/getoutput/<string:wav_name>')
def getoutput(wav_name):
    return Opentrack(wav_name)

# <--- return harmonica error version and second --->
@app.route('/sendsec')
def send():
    sec = {
            'start': 3, 
            'end': 5,
            'drag': False,
            'resize': False,
            'type': 0,
        }
    return jsonify(sec)

# <--- front end upload file to local dir --->
@app.route('/upload', methods=['POST'])
def fileUpload():
    target = os.path.join(UPLOAD_FOLDER, 'sound')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file'] 
    filename = secure_filename(file.filename)
    destination = "/".join([target, filename])
    file.save(destination)
    session['uploadFilePath'] = destination
    response = "Whatever you wish too return"
    return response

@app.route('/wav/<wav_name>')
def streamwav(wav_name):
    def generate():
        with open(app.config["CLIENT_WAV"] + str(wav_name), "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/wav")


if __name__ == '__main__':
    app.secret_key = os.urandom(24)
    app.debug = True
    app.run(host="192.168.50.225")    

