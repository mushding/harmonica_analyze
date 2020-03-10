from flask import Flask,redirect
from flask import url_for

app = Flask(__name__)

@app.route('/')
def index():
    return 'hello man'

@app.route('/user/<username>')
def username(username):
    return 'i am ' + username

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
    app.run()

