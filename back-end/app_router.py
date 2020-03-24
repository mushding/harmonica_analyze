from flask import Flask, redirect, request, url_for, jsonify, session
from flask import app, Response, send_file
from flask_cors import CORS
import requests
import os 
import app_backend

# <--- start flask --->
app = Flask(__name__)
CORS(app)

# <--- app config area --->


@app.route('/url')
def url():
    requests.get("http://192.168.50.225:5001")
    return "TEST"

if __name__ == '__main__':
    app.secret_key = os.urandom(24)
    app.debug = True
    app.run(host="0.0.0.0", port=5000)    

