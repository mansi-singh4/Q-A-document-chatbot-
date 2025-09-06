import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Parameters
CHUNK_SIZE = 500  # characters
INDEX_FILE = "vector_index.faiss"
MAPPING_FILE = "doc_mapping.pkl"

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def read_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def build_index(pdf_path):
    print(f"Processing {pdf_path}...")
    text = read_pdf(pdf_path)
    if not text.strip():
        print("No extractable text found in PDF.")
        return
    
    chunks = chunk_text(text)
    if not chunks:
        print(" No chunks created.")
        return
    
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, INDEX_FILE)
    print("Index built and saved!")


if __name__ == "__main__":
    pdf_file = input("Enter PDF file path: ")
    build_index(pdf_file)

