# 📚 Multi-Session PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload one or more PDF documents and chat with their content using Large Language Models (LLMs). The application supports multiple independent chat sessions, maintains conversation history, and retrieves relevant document chunks to generate accurate answers.

---

## 🚀 Features

* 📄 Upload one or multiple PDF documents
* 🤖 Chat with your PDFs using Retrieval-Augmented Generation (RAG)
* 🧠 Session-based memory for independent conversations
* 💬 Multi-session support with separate vector stores
* 🔍 Semantic search using ChromaDB
* ⚡ Fast LLM inference with Groq (Llama 3.1 8B)
* 🎨 Interactive Streamlit interface
* 🗑️ Delete and reset chat sessions

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* Groq API
* Llama 3.1 8B Instant
* FAISS / Vector Embeddings
* python-dotenv

---

## 📂 Project Structure

```
project2_genAI_RAG/
│
├── app.py                 # Streamlit application
├── requirements.txt
├── .gitignore
│
├── engines/
│   ├── base.py
│   ├── rag_engine.py
│   └── simple_faq.py
│
├── rag/
│   ├── ingest.py
│   └── pipeline.py
│
└── chroma/                # Generated vector database (ignored by Git)
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/project2_genAI_RAG.git
cd project2_genAI_RAG
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📖 How It Works

1. Enter a unique Session ID.
2. Upload one or more PDF files.
3. Documents are split into chunks.
4. Chunks are embedded and stored in ChromaDB.
5. The retriever searches for the most relevant chunks.
6. The LLM generates answers using the retrieved context.
7. Conversation history is maintained separately for each session.

---

## 📸 Demo

Add screenshots or a GIF here showing:

* PDF upload
* Chat interface
* Multiple sessions

---

## 🔮 Future Improvements

* Support additional document formats (DOCX, TXT)
* User authentication
* Conversation export
* Streaming LLM responses
* Cloud deployment
* Citation highlighting

---

## 👩‍💻 Author

**Farah Amr Mohamed Shawky**

Artificial Intelligence & Software Engineering Enthusiast

---
