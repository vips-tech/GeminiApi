from flask import Flask, request, jsonify
from PIL import Image
from google import genai
import os
import json
import re

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
You are an expert plant pathologist. Analyze the plant disease in this image.

Provide a response in EXACTLY this JSON format (no markdown, no extra text):

{
  "disease_name": "Common name (Scientific name)",
  "how_it_occurs": [
    "cause/condition 1",
    "cause/condition 2",
    "cause/condition 3"
  ],
  "treatment": [
    "treatment method 1",
    "treatment method 2",
    "treatment method 3"
  ],
  "precautions": [
    "prevention tip 1",
    "prevention tip 2",
    "prevention tip 3"
  ]
}

Keep each point concise (max 10-15 words). Focus on actionable information only.
If no disease is detected, return: {"disease_name": "No disease detected", "how_it_occurs": [], "treatment": [], "precautions": []}
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
        
        # Try to parse the JSON response from Gemini
        try:
            # Remove markdown code blocks if present
            cleaned_result = re.sub(r'```json\s*|\s*```', '', result).strip()
            analysis_data = json.loads(cleaned_result)
            
            return jsonify({
                "status": "success",
                "disease_name": analysis_data.get("disease_name", "Unknown"),
                "how_it_occurs": analysis_data.get("how_it_occurs", []),
                "treatment": analysis_data.get("treatment", []),
                "precautions": analysis_data.get("precautions", [])
            })
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw text
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
