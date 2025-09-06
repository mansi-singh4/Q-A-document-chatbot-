# backend
import os
from dotenv import load_dotenv
import io
import PyPDF2
import wikipedia
import google.generativeai as genai
import chromadb
import streamlit as st
from notion_client import Client as NotionClient

# Loading config & configuring genai here
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print("Warning: genai.configure failed:", e)
else:
    print("Warning: GOOGLE_API_KEY not set. genai calls will fail.")

# Initializing chromaDB in session state
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()

if "collection" not in st.session_state:
    try:
        st.session_state.collection = st.session_state.chroma_client.get_collection(name="pdf_docs")
    except Exception:
        st.session_state.collection = st.session_state.chroma_client.create_collection(name="pdf_docs")

# debug counters
if "collection_docs_count" not in st.session_state:
    st.session_state.collection_docs_count = 0
if "last_chunks_sample" not in st.session_state:
    st.session_state.last_chunks_sample = []


# Embedding
def get_embeddings(text: str):
    """Generate embeddings using Google Generative AI."""
    try:
        model = "models/embedding-001"
        response = genai.embed_content(model=model, content=text, task_type="retrieval_document")
        emb = response.get("embedding") if isinstance(response, dict) else response
        return emb
    except Exception as e:
        print("Error getting embeddings:", e)
        return None


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into overlapping chunks for embedding."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - chunk_overlap
    return chunks


def reset_collection():
    """Clear and reinitialize the ChromaDB collection (DO NOT clear chat messages)."""
    try:
        st.session_state.chroma_client.delete_collection(name="pdf_docs")
    except Exception:
        pass
    st.session_state.collection = st.session_state.chroma_client.create_collection(name="pdf_docs")
    st.session_state.collection_docs_count = 0
    st.session_state.last_chunks_sample = []


# DB Add Helper
def add_chunks_to_db(chunks):
    chunk_ids, chunk_embeddings = [], []
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        embedding = get_embeddings(chunk)
        if embedding:
            chunk_ids.append(chunk_id)
            chunk_embeddings.append(embedding)

    if chunk_ids:
        st.session_state.collection.add(
            documents=chunks,
            embeddings=chunk_embeddings,
            ids=chunk_ids
        )
        # updating debug counters
        st.session_state.collection_docs_count = len(chunk_ids)
        st.session_state.last_chunks_sample = chunks[:3]
    else:
        st.session_state.collection_docs_count = 0
        st.session_state.last_chunks_sample = []


# PDF Upload
def handle_pdf_upload(uploaded_file):
    if uploaded_file is None:
        return "No file provided."

    try:
        bytes_data = uploaded_file.getvalue()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(bytes_data))
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

        reset_collection()
        chunks = get_text_chunks(text)
        if not chunks:
            return " No text extracted from PDF."

        add_chunks_to_db(chunks)
        return f"Successfully processed {len(chunks)} PDF chunks. (embedded: {st.session_state.collection_docs_count})"

    except Exception as e:
        print("PDF processing error:", e)
        return f" Error processing PDF: {e}"

#  Wikipedia Input
def handle_wikipedia_input(title: str, use_summary: bool = False, summary_sentences: int = 6):
    try:
        if not title or not title.strip():
            return "Please enter a Wikipedia page title."

        if use_summary:
            text = wikipedia.summary(title, sentences=summary_sentences, auto_suggest=False, redirect=True)
        else:
            page = wikipedia.page(title, auto_suggest=False, redirect=True)
            text = page.content or ""

        if not text:
            return f" Wikipedia returned no text for '{title}'."

        reset_collection()
        chunks = get_text_chunks(text)
        add_chunks_to_db(chunks)
        return f" Successfully processed {len(chunks)} Wikipedia chunks from '{title}'. (embedded: {st.session_state.collection_docs_count})"

    except wikipedia.exceptions.PageError:
        # try nearest title
        search_results = wikipedia.search(title)
        if search_results:
            nearest_title = search_results[0]
            return f"Page '{title}' not found. Using nearest match '{nearest_title}'.\n" + \
                   handle_wikipedia_input(nearest_title, use_summary, summary_sentences)
        else:
            return f" Wikipedia page not found for: {title}"

    except wikipedia.exceptions.DisambiguationError as e:
        # fallback: pick first option from disambiguation list
        if e.options:
            nearest_title = e.options[0]
            return f" Title '{title}' is ambiguous. Using first option '{nearest_title}'.\n" + \
                   handle_wikipedia_input(nearest_title, use_summary, summary_sentences)
        else:
            return f" Ambiguous title '{title}', but no options found."

    except Exception as e:
        print("Wikipedia fetch error:", e)
        return f"Error fetching Wikipedia page: {e}"

# Notion Input
def handle_notion_input(page_url: str, notion_api_key: str):
    try:
        if not page_url or not notion_api_key:
            return " Provide both Notion page URL and API key."

        notion = NotionClient(auth=notion_api_key)
        page_id = page_url.split("-")[-1]
        blocks = notion.blocks.children.list(page_id)["results"]

        text = ""
        for block in blocks:
            if "paragraph" in block and block["paragraph"]["text"]:
                text += "".join([t.get("plain_text", "") for t in block["paragraph"]["text"]]) + "\n"

        if not text:
            return " No text found on the Notion page."

        reset_collection()
        chunks = get_text_chunks(text)
        add_chunks_to_db(chunks)
        return f" Successfully processed {len(chunks)} Notion chunks. (embedded: {st.session_state.collection_docs_count})"
    except Exception as e:
        print("Notion fetch error:", e)
        return f" Error fetching Notion page: {e}"



# Answer Generator
def get_answer(question: str):
    try:
        if not question or not question.strip():
            return " Please type a question."

      
        query_embedding = get_embeddings(question)
        if not query_embedding:
            return " Failed to generate embedding for the question."

       
        results = st.session_state.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
        )

       
        docs = results.get("documents") if isinstance(results, dict) else (results.documents if hasattr(results, "documents") else None)
        context = ""
        if docs:
           
            first = docs[0]
            if isinstance(first, list):
                context = " ".join(first)
            elif isinstance(first, str):
                context = first

       
        if not context:
            return " I don't have enough context. Please upload a PDF or fetch a source first."

       #prompting the model
        prompt_text = f"""
Using only the following context, answer the question. If the answer is not in the context, say you can't find the answer.

Context:
{context}

Question: {question}

Answer:
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt_text)
        # response shape — be defensive
        ans = response.text if hasattr(response, "text") else (response.get("candidates")[0].get("content") if isinstance(response, dict) else str(response))
        return ans

    except Exception as e:
        print("Get answer error:", e)
        return f"Error while generating answer: {e}"
