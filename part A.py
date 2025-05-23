# part_a_model.py

import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load dataset
df = pd.read_csv('/Users/vengadesh/text-similarity-api/DataNeuron_Text_Similarity.csv')

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode both sets of texts
embeddings1 = model.encode(df['text1'].tolist(), convert_to_tensor=True, show_progress_bar=True)
embeddings2 = model.encode(df['text2'].tolist(), convert_to_tensor=True, show_progress_bar=True)

# Compute cosine similarity
similarity_scores = util.cos_sim(embeddings1, embeddings2).diagonal()

# Add similarity scores to DataFrame
df['similarity_score'] = similarity_scores.cpu().numpy()

# Save the result
df.to_csv('Text_Similarity_Scored.csv', index=False)

print("Similarity scores saved to Text_Similarity_Scored.csv")
