from flask import Flask, redirect,request
from flask import url_for
from flask import render_template
from flask import send_file, send_from_directory,safe_join,abort


app = Flask(__name__)

app.config["CLIENT_WAV"] = "/home/micro/harmonica_train/harmonica_project/back-end/static/HarmonicaData/wav"



@app.route('/loginurl',methods = ['GET','POST'])
def login():
    if request.method =='POST':
        return redirect(url_for('hello',username = request.form.get('username')))

    return render_template('login.html')

@app.route('/get-wav/<wav_name>')
def get_wav(wav_name):
    try:
        return send_from_directory(app.config["CLIENT_WAV"], filename = wav_name, as_attachment = True)

    except FileNotFoundError:
        abort(404)





@app.route('/para/<user>')
def index(user):
    return  render_template('abc.html',user_template=user)


@app.route('/hello/<username>')
def hello(username):
    return render_template('hello.html',username = username)



@app.route('/age/<int:age>')
def userage(age):
    return 'i am ' + str(age) + ' years old'  

@app.route('/hight/<float:hight>')
def userhight(hight):
    return str(hight) + ' cm.'

@app.route('/a')
def url_for_a():
    return 'here is a'

@app.route('/b')
def b():
    return redirect(url_for('url_for_a')) 

    
if __name__ == '__main__':
    app.debug = True
    app.run(host="192.168.50.225")

