import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and chunks
index = faiss.read_index("vector_index.faiss")
with open("doc_mapping.pkl", "rb") as f:
    chunks = pickle.load(f)

def query_index(question, top_k=3):
    q_vec = embedder.encode([question], convert_to_tensor=False)
    q_vec = np.array(q_vec).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    return [chunks[i] for i in indices[0]]

# Example usage
question = input("Enter your question: ")
results = query_index(question)
print("\nTop relevant chunks:\n")
for i, chunk in enumerate(results):
    print(f"Chunk {i+1}: {chunk[:500]}...\n")
