# backend.py
import os
import io
import PyPDF2
import wikipedia
import google.generativeai as genai
import chromadb
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not set. GenAI calls will fail.")

# Initialize ChromaDB collection in session state
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()

if "collection" not in st.session_state:
    try:
        st.session_state.collection = st.session_state.chroma_client.get_collection(name="pdf_docs")
    except Exception:
        st.session_state.collection = st.session_state.chroma_client.create_collection(name="pdf_docs")

# Helper: generate embeddings
def get_embeddings(text: str):
    try:
        model = "models/embedding-001"
        response = genai.embed_content(model=model, content=text, task_type="retrieval_document")
        return response.get("embedding")
    except Exception as e:
        print("Error getting embeddings:", e)
        return None

# Helper: split text into chunks
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - chunk_overlap
    return chunks

# Helper: reset DB collection
def reset_collection():
    try:
        st.session_state.chroma_client.delete_collection(name="pdf_docs")
    except Exception:
        pass
    st.session_state.collection = st.session_state.chroma_client.create_collection(name="pdf_docs")

# Helper: add chunks to DB
def add_chunks_to_db(chunks):
    chunk_ids, chunk_embeddings = [], []
    for i, chunk in enumerate(chunks):
        emb = get_embeddings(chunk)
        if emb:
            chunk_ids.append(f"chunk_{i}")
            chunk_embeddings.append(emb)
    if chunk_ids:
        st.session_state.collection.add(
            documents=chunks,
            embeddings=chunk_embeddings,
            ids=chunk_ids
        )

# Handle PDF uploads
def handle_pdf_upload(uploaded_file):
    if not uploaded_file:
        return "No PDF file provided."

    try:
        bytes_data = uploaded_file.getvalue()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(bytes_data))
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

        if not text.strip():
            return "No text extracted from PDF."

        reset_collection()
        chunks = get_text_chunks(text)
        add_chunks_to_db(chunks)

        return f"✅ Successfully processed {len(chunks)} PDF chunks."

    except Exception as e:
        return f"Error processing PDF: {e}"

# Handle Wikipedia page fetching
def handle_wikipedia_input(title: str, use_summary=False, summary_sentences=6):
    if not title or not title.strip():
        return "Please enter a Wikipedia page title."

    try:
        if use_summary:
            text = wikipedia.summary(title, sentences=summary_sentences, auto_suggest=False, redirect=True)
        else:
            page = wikipedia.page(title, auto_suggest=False, redirect=True)
            text = page.content or ""

        if not text.strip():
            return f"No text returned from Wikipedia for '{title}'."

        reset_collection()
        chunks = get_text_chunks(text)
        add_chunks_to_db(chunks)
        return f"✅ Successfully processed {len(chunks)} Wikipedia chunks from '{title}'."

    except wikipedia.exceptions.PageError:
        search_results = wikipedia.search(title)
        if search_results:
            nearest_title = search_results[0]
            return f"Page '{title}' not found. Using nearest match '{nearest_title}'.\n" + \
                   handle_wikipedia_input(nearest_title, use_summary, summary_sentences)
        else:
            return f"Wikipedia page not found for: {title}"

    except wikipedia.exceptions.DisambiguationError as e:
        if e.options:
            nearest_title = e.options[0]
            return f"Title '{title}' is ambiguous. Using first option '{nearest_title}'.\n" + \
                   handle_wikipedia_input(nearest_title, use_summary, summary_sentences)
        else:
            return f"Ambiguous title '{title}', but no options found."

    except Exception as e:
        return f"Error fetching Wikipedia page: {e}"

# Generate answer from embedded context
def get_answer(question: str):
    if not question.strip():
        return "Please type a question."

    try:
        query_embedding = get_embeddings(question)
        if not query_embedding:
            return "Failed to generate embedding for the question."

        results = st.session_state.collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        docs = results.get("documents") if isinstance(results, dict) else None
        context = " ".join(docs[0]) if docs and docs[0] else ""
        if not context:
            return "I don't have enough context. Please upload a PDF or fetch a Wikipedia page first."

        prompt_text = f"""
Using only the following context, answer the question. If the answer is not in the context, say you can't find the answer.

Context:
{context}

Question: {question}

Answer:
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt_text)
        return response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        return f"Error generating answer: {e}"
