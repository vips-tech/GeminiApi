from flask import Flask, request, jsonify
from PIL import Image
from google import genai
import os

app = Flask(__name__)

# Replace with your Gemini API key
client = genai.Client(
    api_key="AIzaSyAASQfVIUoxDzWvYmeYkKLOLn2nwdFCKN8"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def analyze_plant_disease(image_path):

    img = Image.open(image_path)

    prompt = """
You are an expert plant pathologist.

Analyze the plant image and provide ONLY the following essential information in a concise format:

1. Disease Name (scientific and common name)
2. How It Occurs (brief explanation of causes and favorable conditions)
3. How to Solve (2-3 key treatment methods)
4. Precautions (2-3 important prevention measures)

Keep each section brief and to the point. Use bullet points for clarity.
Avoid lengthy descriptions - focus only on actionable information.

Format your response as:

Disease Name: [name]

How It Occurs:
- [cause 1]
- [cause 2]

How to Solve:
- [treatment 1]
- [treatment 2]

Precautions:
- [prevention 1]
- [prevention 2]

If the plant appears healthy, simply state "No disease detected."
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img]
    )

    return response.text


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Plant Disease Prediction API"
    })


@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "error": "No file selected"
            }), 400

        filepath = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(filepath)

        result = analyze_plant_disease(filepath)

        return jsonify({
            "status": "success",
            "analysis": result
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
