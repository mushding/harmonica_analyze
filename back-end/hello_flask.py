from flask import Flask, redirect,request,flash,app
from flask import url_for,Response
from flask import render_template
from flask import send_file, send_from_directory,safe_join,abort
from scipy.io import wavfile

app = Flask(__name__)

app.config["CLIENT_WAV"] = "/home/micro/harmonica_train/harmonica_project/back-end/static/HarmonicaData/wav/"


@app.route('/')
def home():
    return 'HOME'

@app.route('/loginurl',methods = ['GET','POST'])
def login():
    if request.method =='POST':
        if login_check(request.form['username'], request.form['password']): 
            flash('Login Success!')
            return redirect(url_for('hello',username = request.form.get('username')))

    return render_template('login.html')

def login_check(username, password):
    """登入帳密檢核"""
    if username == 'chris' and password == 'hello':
        return True
    else:
        return False

@app.route('/hello/<username>')
def hello(username):
    return render_template('hello.html',username = username)

@app.route('/get-wav/<wav_name>')
def get_wav(wav_name):
    try:
        return app.send_static_file(app.config["CLIENT_WAV"] + str(wav_name))

    except FileNotFoundError:
        abort(404)


@app.route('/wav/<wav_name>')
def streamwav(wav_name):
    def generate():
        with open(app.config["CLIENT_WAV"] + str(wav_name), "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/wav")

@app.route('/test/<string:wav_name>', methods=['GET'])
def test(wav_name):
    path_to_file = app.config["CLIENT_WAV"] + wav_name
    return send_file(
         path_to_file, 
         mimetype="audio/wav", 
         as_attachment=True, 
         attachment_filename=wav_name)
    
if __name__ == '__main__':
    app.debug = True
    app.run(host="192.168.50.225")

