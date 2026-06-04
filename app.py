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

Analyze the plant image and provide:

1. Plant Species
2. Disease Detected
3. Severity Level
4. Symptoms Observed
5. Possible Causes
6. Treatment Recommendations
7. Prevention Tips

Format:

Plant Species:
Disease Detected:
Severity Level:
Symptoms Observed:
Possible Causes:
Treatment Recommendations:
Prevention Tips:

If the plant appears healthy, clearly mention that no disease is detected.
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
