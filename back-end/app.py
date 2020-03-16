from flask import Flask, redirect, request, url_for, jsonify, session
from flask import app, Response, send_file
from flask_cors import CORS
from open_track import Open_track
from load_model import CNN
from werkzeug.utils import secure_filename
import os 

# <--- start flask --->
app = Flask(__name__)
CORS(app)

# <--- app config area --->
app.config["CLIENT_WAV"] = "../back-end/static/HarmonicaData/wav"

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

# <--- front end upload file to local dir --->
@app.route('/upload', methods=['POST'])
def fileUpload():
    if not os.path.isdir(app.config["CLIENT_WAV"]):
        os.mkdir(app.config["CLIENT_WAV"])
    file = request.files['file'] 
    filename = secure_filename(file.filename)
    destination = "/".join([app.config["CLIENT_WAV"], filename])
    file.save(destination)
    session['uploadFilePath'] = destination
    response = "Whatever you wish too return"
    return response

@app.route('/wav/<wav_name>')
def streamwav(wav_name):
    def generate():
        target = os.path.join(app.config["CLIENT_WAV"], str(wav_name))
        with open(target, "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/wav")


if __name__ == '__main__':
    app.secret_key = os.urandom(24)
    app.debug = True
    app.run(host="192.168.50.225")    

