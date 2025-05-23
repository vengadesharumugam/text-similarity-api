# app.py

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optional: Load cloud API keys / URLs if needed
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY")  # Example
CLOUD_API_URL = os.getenv("CLOUD_API_URL")  # Example

# Initialize Flask app and Sentence-BERT model
app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/similarity', methods=['POST'])
def compute_similarity():
    data = request.get_json()

    # Validate required fields
    if not data or 'text1' not in data or 'text2' not in data:
        return jsonify({"error": "Both 'text1' and 'text2' are required."}), 400

    text1 = data['text1']
    text2 = data['text2']

    if not text1.strip() or not text2.strip():
        return jsonify({"error": "Input texts must not be empty."}), 400

    # Encode both texts using the sentence transformer
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.cos_sim(embedding1, embedding2).item()

    # Return similarity score
    return jsonify({"similarity score": round(similarity_score, 4)})

if __name__ == '__main__':
    # For local development; on Render/Heroku this is handled by Procfile
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
