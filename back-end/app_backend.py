import sys
from flask import Flask, redirect, request, url_for, jsonify, session
from flask import app, Response, send_file, render_template
from flask_cors import CORS
from open_track import Open_track
from load_model import CNN
from werkzeug.utils import secure_filename
import subprocess
import os
from note_predict.note_compare import Note_compare

# <--- start flask --->
app = Flask(__name__, template_folder="templates",
            static_folder="static", static_url_path="/back-end/static")

# <--- app config area --->
app.config["CLIENT_WAV"] = "../back-end/static/HarmonicaData/wav"

@app.route('/tests')
def tests():
    compare = Note_compare()
    compare.generate_correct_user_timeline()
    return "tests"

@app.route('/<path:path>')
def catch_all(path):
    return 'You want path: %s' % path


@app.route('/')
def home():
    return render_template('index.html')


def Opentrack(wav_name):
    open = Open_track()
    tensor_track = open.readsound(wav_name)

    model = open.loadmodel()

    output = open.putincnn(tensor_track, model)

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
    print(request.files)
    file = request.files['file']
    filename = secure_filename(file.filename)
    destination = "/".join([app.config["CLIENT_WAV"], filename])
    file.save(destination)
    session['uploadFilePath'] = destination
    response = "Whatever you wish too return"
    return response

# <--- front end record blob file to backend --->
@app.route('/recordUpload', methods=['POST'])
def recordUpload():
    if not os.path.isdir(app.config["CLIENT_WAV"]):
        os.mkdir(app.config["CLIENT_WAV"])
    file = request.files['data']
    filename = secure_filename(file.filename)
    destination = "/".join([app.config["CLIENT_WAV"], filename])
    file.save(destination)
    record_destination = "/".join([app.config["CLIENT_WAV"], "record.wav"])
    command = "ffmpeg -y -i " + destination + " -ab 160k -ac 2 -ar 44100 -vn " + record_destination
    subprocess.call(command, shell=True)
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

    app.run(host="0.0.0.0", port=80)
