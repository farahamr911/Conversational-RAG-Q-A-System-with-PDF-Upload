import streamlit as st
import os, tempfile
from rag.ingest import load_and_split_pdf, create_vector_store
from rag.pipeline import pipeline
from engines.rag_engine import RagEngine
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage


# ------------------------------
# Streamlit page config
# ------------------------------
st.set_page_config(page_title="📚 PDF Chatbot", layout="wide")

# ------------------------------
# Sidebar: Session & Bot Avatar
# ------------------------------
st.sidebar.title("Settings")
Setting = st.sidebar.selectbox("Choose Option", ["Enter Session ID", "CHAT BOT"])
bot_avatar = st.sidebar.selectbox(
    "Bot Avatar", ["🤖", "💡", "🦄", "👩‍💻", "🧑‍🚀", "🐼"], index=0
)

# Multi-session storage
if "sessions" not in st.session_state:
    st.session_state.sessions = {}  # { session_id: { "chat_history": [], "db": ..., "rag_pipeline": ..., "rag_engine": ... } }

# ------------------------------
# Get session ID
# ------------------------------


# Sidebar: Session ID input
if Setting == "Enter Session ID":
    if "session_id" not in st.session_state:
        st.session_state.session_id = ""
    st.session_state.session_id = st.sidebar.text_input(
        "Enter Session ID",
        value=st.session_state.session_id,
        key="session_id_input"
    )
else:  # CHAT BOT
    if "session_id" not in st.session_state:
        st.session_state.session_id = ""
    st.session_state.session_id = st.sidebar.text_input(
        "Session ID for CHAT BOT",
        value=st.session_state.session_id,
        key="chatbot_session_id_input"
    )

session_id = st.session_state.session_id

if not session_id:
    st.warning("Please enter a Session ID to start chatting.")
    st.stop()


# Initialize session if not exists
if session_id not in st.session_state.sessions:
    st.session_state.sessions[session_id] = {
        "chat_history": [],
        "db": None,
        "rag_pipeline": None,
        "rag_engine": None
    }

session_data = st.session_state.sessions[session_id]

# ------------------------------
# Load LLM (once)
# ------------------------------
if "llm" not in st.session_state:
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    st.session_state.llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant")
    
llm = st.session_state.llm

# ------------------------------
# PDF Upload for this session
# ------------------------------
st.header(f"📄 Upload PDFs (Session: {session_id})")
uploaded_files = st.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True, key=f"upload_{session_id}"
)

if uploaded_files and session_data["db"] is None:
    temp_dir = tempfile.mkdtemp()
    pdf_paths = []
    for f in uploaded_files:
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.read())
        pdf_paths.append(path)

    docs = load_and_split_pdf(pdf_paths)
    if docs:
        session_data["db"] = create_vector_store(docs, session_id)
        st.success("✅ PDFs processed and vector store ready!")

# ------------------------------
# Initialize RAG pipeline & engine for this session
# ------------------------------
if session_data["db"] and session_data["rag_pipeline"] is None:
    session_data["rag_pipeline"] = pipeline(session_data["db"], llm)
    session_data["rag_engine"] = RagEngine(
        session_data["rag_pipeline"].build_rag_pipeline(session_id=session_id),
        session_data["rag_pipeline"].history_store
    )

# ------------------------------
# Chat interface
# ------------------------------
st.header(f"💬 Chat with your PDFs (Session: {session_id})")

# Display chat history
for message in session_data["chat_history"]:
    role = "user" if message["role"] == "user" else "assistant"
    avatar = bot_avatar if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

# User input
if session_data["db"]:
    delete = st.sidebar.button("Delete Session", key=f"delete_{session_id}")
    user_input = st.chat_input("Ask a question...", key=f"user_input_{session_id}")
    
    if user_input:
        # Add user message
        session_data["chat_history"].append({"role": "user", "content": user_input})
        session_data["rag_pipeline"].add_user_message(HumanMessage(content=user_input), session_id=session_id)

        # Get assistant answer
        answer = session_data["rag_engine"].answer(session_id=session_id, question=user_input)
        answer_text = answer.content

        # Add assistant message
        session_data["chat_history"].append({"role": "assistant", "content": answer_text})
        session_data["rag_pipeline"].add_AI_message(AIMessage(content=answer_text), session_id=session_id)

        # Display new messages
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant", avatar=bot_avatar):
            st.markdown(answer_text)
    
    if (delete):
        session_data["chat_history"]=[]
        session_data["db"]=None
        session_data["rag_pipeline"]=None
        session_data["rag_engine"]= None
        
else:
    st.warning("Please upload PDFs to start chatting for this session.")