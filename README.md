# text-similarity-api
Part A: Semantic Similarity Model
---------------------------------
Approach:
- I used the 'all-MiniLM-L6-v2' model from the SentenceTransformers library to compute sentence
embeddings.
- For each pair of text paragraphs, the cosine similarity between their embeddings was computed to
generate a similarity score between 0 and 1.
- This method captures semantic meaning effectively and works well in unsupervised settings.
Part B: Deployment
------------------
- A Flask-based REST API was developed to expose the model.
- The API accepts POST requests with JSON input in the format:
{"text1": "...", "text2": "..."}
- It returns a similarity score in the response:
{"similarity score": 0.85}
- The API is deployed on Heroku for live access.
Tools Used:
- Python
- SentenceTransformers
- Flask
- Heroku for deployment
