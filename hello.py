from flask import Flask, request
from chunking import chunk_text;
from flask_cors import CORS




app = Flask(__name__)

CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/processText", methods=["POST"])
def process_text():
    data = request.get_json()
    text = data['text']
    processed =chunk_text(text)
    for chunk in processed:
        print(chunk)
        print("\n\n")
    return processed