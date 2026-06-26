# This module handles PDF ingestion: loading, chunking, embedding, and storing in Chroma.
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader #LangChain internally uses: import pypdf
#pypdf	Low-level PDF reader used internally by LangChain
#langchain-community	Provides ready-made loaders built on top of pypdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List, Optional
import tempfile


class Ingest_config:     #for intializationthe ingest_config
    def __init__(self):
        self.chunk_size=500
        self.overlap=100
        self.embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.persist_directory=r"C:\Users\farah\Downloads\chroma test rag"

config = Ingest_config()

def load_and_split_pdf(pdf_paths: List[str])->list:
    documents=[]
    for path in pdf_paths:
        try:
            loader= PyPDFLoader(path)   #PyPDFLoader bia5od file path laken PyPDFDirectoryLoader bia5od folder w y load kol al files ali fiha 
            doc=loader.load()      #load bete3mel kol page ka2enha document
            documents.extend(doc) #3ashan a3raf kol page taba3 anhy pdf file
        except Exception as e:
            st.error(f"Error loading PDF {path}: {str(e)}")
    if not documents:
        st.error("No documents were loaded. Please check your PDF files.")
        return []
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=config.chunk_size,chunk_overlap=config.overlap)
    doc_splitted=text_splitter.split_documents(documents)
    return doc_splitted

def create_vector_store(documents,session_id:str)->Chroma:
    if not documents:
       st.error("No documents to process. Skipping vector store creation.")
       return None
    # Format the persist_directory with the actual session_id
    persist_dir = config.persist_directory.format(session_id=session_id)
    #persist_dir=r"C:\Users\farah\Downloads\test_zeft"
    embeddings=HuggingFaceEmbeddings(model_name=config.embedding_model)
    db = Chroma.from_documents(documents, embedding=embeddings,persist_directory=persist_dir)
    st.success("Vector store created/updated successfully.")
    return db
