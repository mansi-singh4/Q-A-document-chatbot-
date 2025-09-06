#Frontend
import os
import io
import streamlit as st
import PyPDF2
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv


# Config & Setup
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="AI Chatbot", page_icon="📄", layout="wide")


# CSS Styling 
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e2f;
        color: #f5f5f5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main {
        background-color: #2a2d3e;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.4);
    }
    .chat-bubble {
        padding: 14px;
        border-radius: 14px;
        margin: 10px 0;
        max-width: 75%;
        word-wrap: break-word;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 15px;
    }
    .user-bubble {
        background-color: #5865f2;
        color: white;
        text-align: right;
        margin-left: auto;
        border-top-right-radius: 2px;
    }
    .assistant-bubble {
        background-color: #40444b;
        color: #ffffff;
        text-align: left;
        margin-right: auto;
        border-top-left-radius: 2px;
    }
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
    }
    .header-title {
        font-size: 32px;
        font-weight: bold;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 15px;'>
        <div style='
            background-color: #5865f2;
            color: white;
            font-size: 22px;
            font-weight: bold;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        '>
            m&m
        </div>
        <div class='header-title'>Turning your Pages into the Answers you Seek</div>
    </div>
    """,
    unsafe_allow_html=True
)



# Session State
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()

if "collection" not in st.session_state:
    try:
        st.session_state.collection = st.session_state.chroma_client.get_collection(name="pdf_docs")
    except Exception:
        st.session_state.collection = st.session_state.chroma_client.create_collection(name="pdf_docs")

if "messages" not in st.session_state:
    st.session_state.messages = []



# Helper Functions
def get_embeddings(text):
    try:
        model = 'models/embedding-001'
        response = genai.embed_content(model=model, content=text, task_type="retrieval_document")
        return response['embedding']
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return None


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    if not text:
        return chunks
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def handle_pdf_upload(uploaded_file):
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(bytes_data))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # resetting the db and msgs
            st.session_state.chroma_client.delete_collection(name="pdf_docs")
            st.session_state.collection = st.session_state.chroma_client.create_collection(name="pdf_docs")
            st.session_state.messages = []
            
            chunks = get_text_chunks(text)
            
            with st.spinner(f"Processing {len(chunks)} text chunks..."):
                chunk_ids, chunk_embeddings = [], []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"chunk_{i}"
                    embedding = get_embeddings(chunk)
                    if embedding:
                        chunk_ids.append(chunk_id)
                        chunk_embeddings.append(embedding)
            
                st.session_state.collection.add(
                    documents=chunks,
                    embeddings=chunk_embeddings,
                    ids=chunk_ids
                )
            
            st.success(f"✅ Successfully processed {len(chunks)} text chunks.")
            st.session_state.messages.append({"role": "assistant", "content": "📄 Document uploaded! Ask me a question about it."})
            
        except Exception as e:
            st.error(f"An error occurred during file processing: {e}")

#UI of the app 
st.markdown(
    "<div class='header-title'>🤖 AI Knowledge ChatBot</div>",
    unsafe_allow_html=True
)
st.markdown("_Upload a PDF and start chatting with it!_")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/pdf.png", width=120)
    st.subheader("📂 Input Options")
    
    # PDF Upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf")
    if uploaded_file:
        from backend import handle_pdf_upload
        st.info(handle_pdf_upload(uploaded_file))
    
    # Wikipedia Input
    wiki_title = st.text_input("🌍 Wikipedia Page Title")
    if st.button("Fetch Wikipedia"):
        from backend import handle_wikipedia_input
        st.info(handle_wikipedia_input(wiki_title))
    
    # Notion Input
    notion_url = st.text_input("📝 Notion Page URL")
    notion_api_key = st.text_input("🔑 Notion API Key", type="password")
    if st.button("Fetch Notion"):
        from backend import handle_notion_input
        st.info(handle_notion_input(notion_url, notion_api_key))
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
    st.caption("💡 Tip: Upload a PDF, Wikipedia page, or Notion page and ask questions about it.")




#  Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div class='chat-bubble user-bubble'>
                <img src="https://img.icons8.com/ios-filled/50/ffffff/user.png" class="avatar">
                {message['content']}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='chat-bubble assistant-bubble'>
                <img src="https://img.icons8.com/color/48/000000/bot.png" class="avatar">
                {message['content']}
            </div>
            """,
            unsafe_allow_html=True,
        )


# Chat Input
if prompt := st.chat_input("💬 Ask something about the PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f"""
        <div class='chat-bubble user-bubble'>
            <img src="https://img.icons8.com/ios-filled/50/ffffff/user.png" class="avatar">
            {prompt}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Hunting for answers in the document..."):
        try:
            query_embedding = get_embeddings(prompt)
            if not query_embedding:
                st.error("Failed to generate embedding for the question.")
                st.stop()

            results = st.session_state.collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
            )
            
            context = ""
            if results["documents"]:
                context = " ".join(results["documents"][0])
            
            if not context:
                answer = "I don't have enough context from the document to answer that question. Please upload a PDF first."
            else:
                prompt_text = f"""
                Using only the following context, answer the question.
                If the answer is not in the context, say you can't find the answer. 

                Context:
                {context}

                Question: {prompt}

                Answer:
                """
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt_text)
                answer = response.text
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(
                f"""
                <div class='chat-bubble assistant-bubble'>
                    <img src="https://img.icons8.com/color/48/000000/bot.png" class="avatar">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True,
            )
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ An error occurred: {e}"})
            

            
