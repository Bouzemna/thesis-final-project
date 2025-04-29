import nltk
import pandas as pd
import re
import faiss
import numpy as np
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# Load dataset
file_path = "/Users/emnabouzguenda/Desktop/telegram_bot_project/cleaned_responses.csv"
df = pd.read_csv(file_path)

# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


# Convert cleaned responses into embeddings
embeddings = model.encode(df['cleaned_response'].tolist(), convert_to_numpy=True)

# Create FAISS index with Inner Product (IP) for cosine similarity

index = faiss.IndexFlatL2(embeddings.shape[1])  # Use L2 instead of IP  # NEW (Force Correct Metric)
index.add(embeddings)  # Add embeddings to index

# Save FAISS index
faiss.write_index(index, "/Users/emnabouzguenda/Desktop/telegram_bot_project/faiss_index.bin")

# Save embeddings and metadata
df['embedding_vector'] = embeddings.tolist()
df.to_csv("/Users/emnabouzguenda/Desktop/telegram_bot_project/embedded_responses.csv", index=False)

print("Embeddings generated and FAISS index saved.")


# Load FAISS index for retrieval
def retrieve_similar_responses(queries, top_k=10):  # Increased top_k for more diverse results
    """Finds the most similar responses for multiple queries and applies reranking."""

    # Load FAISS index
    index = faiss.read_index("/Users/emnabouzguenda/Desktop/telegram_bot_project/faiss_index.bin")


    results = {}
    epsilon = 1e-8  # Small value to prevent division errors in JS Divergence

    for query in queries:
        # Encode and normalize query
        query_embedding = model.encode([query], convert_to_numpy=True)
        query_embedding = normalize(query_embedding, norm='l2')

        # Find most similar responses
        cosine_similarities, indices = index.search(query_embedding, top_k)  # Now returns cosine similarities directly

        # Convert Cosine Similarity to Cosine Distance (1 - Cosine Similarity)
        cosine_distances = 1 - cosine_similarities[0]

        # Compute JS Divergence
        retrieved_embeddings = np.array([df.iloc[idx]['embedding_vector'] for idx in indices[0]])
        query_prob = softmax(query_embedding.flatten() + epsilon)
        retrieved_probs = np.array([softmax(emb + epsilon) for emb in retrieved_embeddings])
        js_divergences = [jensenshannon(query_prob, retrieved_probs[i]) for i in range(len(retrieved_probs))]

        # Retrieve matching responses with similarity scores, removing duplicates and low-quality responses
        seen_responses = set()
        ranked_responses = []

        for i, idx in enumerate(indices[0]):
            response_text = df.iloc[idx]['cleaned_response']
            if response_text not in seen_responses and len(
                    response_text.split()) > 3:  # Filter out very short responses
                seen_responses.add(response_text)
                score = 0.85 * cosine_similarities[0][i] - 0.15 * cosine_distances[i]  # Adjusted ranking formula
                ranked_responses.append({
                    "response": response_text,
                    "cosine_similarity": cosine_similarities[0][i],
                    "cosine_distance": cosine_distances[i],
                    "js_divergence": js_divergences[i],
                    "ranking_score": score
                })

        # Apply reranking with cross-encoder
        # Appliquer le reranking avec le Cross-Encoder
        query_pairs = [[query, r["response"]] for r in ranked_responses]
        print("🚀 Avant Cross-Encoder Prediction")

        # ✅ Vérifier que query_pairs contient bien des données
        if not query_pairs:
            print("⚠️ Aucune paire Query-Response envoyée au Cross-Encoder !")
        else:
            print(f"✅ {len(query_pairs)} paires envoyées au Cross-Encoder")

        for pair in query_pairs[:5]:  # Afficher les 5 premières paires
            print("🔍 Query-Response Pair:", pair)

        # Obtenir les scores bruts du Cross-Encoder
        raw_rerank_scores = cross_encoder.predict(query_pairs)
        # ✅ Vérifier les valeurs AVANT normalisation
        print("🔎 Scores bruts du Reranker avant normalisation :", raw_rerank_scores)

        # Min-Max Scaling pour normaliser entre 0 et 1
        min_score = min(raw_rerank_scores)
        max_score = max(raw_rerank_scores)

        if max_score - min_score > 0:
            rerank_scores = [(s - min_score) / (max_score - min_score) for s in raw_rerank_scores]
        else:
            rerank_scores = [0.5] * len(raw_rerank_scores)  # Valeur neutre si tous les scores sont pareils

        # ✅ Vérifier les valeurs APRÈS normalisation
        print("🔎 Scores normalisés du Reranker :", rerank_scores)

        for i, r in enumerate(ranked_responses):
            r["rerank_score"] = rerank_scores[i]
        # Assigner les scores normalisés aux réponses
        for i, r in enumerate(ranked_responses):
            r["rerank_score"] = rerank_scores[i]

        # Remove responses too similar to the top answer
        final_responses = [ranked_responses[0]]  # Keep the top response
        for r in ranked_responses[1:]:
            if cosine_similarity(
                    np.array([[ranked_responses[0]['cosine_similarity']]]),  # Reshaped to 2D
                    np.array([[r['cosine_similarity']]])  # Reshaped to 2D
            )[0][0] < 0.95:  # Threshold to ensure diversity
                final_responses.append(r)
            if len(final_responses) == 5:
                break  # Keep only top 5 responses

        results[query] = final_responses

    return results


# Select random questions from dataset for evaluation
eval_queries = random.sample(df['question'].dropna().tolist(), 3)  # Ensure no NaN values

# Retrieve responses
similar_responses = retrieve_similar_responses(eval_queries)

# Save evaluation metrics to CSV
metrics_data = []
for query, responses in similar_responses.items():
    for resp in responses:
        metrics_data.append([
            query, resp['response'],
            resp['cosine_similarity'], resp['cosine_distance'], resp['js_divergence'], resp['rerank_score']
        ])

metrics_df = pd.DataFrame(metrics_data,
                          columns=["Query", "Response", "Cosine Similarity", "Cosine Distance", "JS Divergence",
                                   "Rerank Score"])
metrics_df.to_csv("/Users/emnabouzguenda/Desktop/telegram_bot_project/evaluation_metrics.csv", index=False)

# Print results
print("\nEvaluation Metrics Saved: evaluation_metrics.csv")
print(metrics_df.describe())  # Summary statistics for evaluation
print("Number of embeddings in FAISS index:", index.ntotal)

import matplotlib.pyplot as plt

# Visualisation des scores
plt.figure(figsize=(12, 5))
plt.hist(metrics_df["Cosine Similarity"], bins=30, alpha=0.7, label="Cosine Similarity")
plt.hist(metrics_df["Rerank Score"], bins=30, alpha=0.7, label="Rerank Score")
plt.xlabel("Score")
plt.ylabel("Nombre d'occurrences")
plt.legend()
plt.title("Distribution des scores de similarité et de reranking")
plt.show()


print(metrics_df.sort_values("Rerank Score", ascending=False).head(10))


print("Moyenne Cosine Similarity (Top 5):", metrics_df.nlargest(5, 'Rerank Score')["Cosine Similarity"].mean())
print("Moyenne Cosine Similarity (Bas 5):", metrics_df.nsmallest(5, 'Rerank Score')["Cosine Similarity"].mean())
