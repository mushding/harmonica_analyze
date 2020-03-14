from flask import Flask, redirect, request, url_for, jsonify
from flask import app, Response
from flask_cors import CORS
from open_track import Open_track

app = Flask(__name__)
CORS(app)
app.config["CLIENT_WAV"] = "/home/micro/harmonica_train/harmonica_project/back-end/static/HarmonicaData/wav/"

@app.route('/')
def home():
    return 'HOME'

def Opentrack(wav_name):
        tensor_track = Open_track.readsound(wav_name)

        model = Open_track.loadmodel()
        
        output = Open_track.putincnn(tensor_track,model)
        return output


@app.route('/getoutput/<wav_name>')
def getoutput(wav_name):
    return Opentrack(wav_name)

@app.route('/sendsec')
def send():
    sec = {
        'start': 3, 
        'end': 5,
        'type': 0,
    }
    return jsonify(sec)


if __name__ == '__main__':
    app.debug = True
    app.run(host="192.168.50.225")    


