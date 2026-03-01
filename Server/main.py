import os
from flask import Flask, request
from flask_cors import CORS
from google import genai

client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["POST"])
def index():
    data = request.get_json()
    response = client.models.generate_content(
        model="gemma-3-1b-it",
        contents=data.get("prompt")
    )
    return response.text

if __name__ == "__main__":
    app.run(debug=True)
