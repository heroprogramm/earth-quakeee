import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import os
import warnings

# App title
st.set_page_config(page_title="PDF Chatbot", page_icon="📚")

# Initialize models (cached for performance)
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=200)
    return HuggingFacePipeline(pipeline=pipe)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI Elements
st.title("📚 PDF Chatbot")
st.markdown("Upload a PDF and ask questions about its content")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    st.markdown("---")
    st.markdown("ℹ️ This app uses FLAN-T5 model for question answering")

# Process PDF and create vector store
def process_pdf(pdf_file):
    with st.spinner("Processing PDF..."):
        try:
            # Save uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Read PDF
            with fitz.open("temp.pdf") as doc:
                text = "".join([page.get_text() for page in doc])
            
            if not text.strip():
                st.error("PDF appears to be empty or contains no text")
                return None
            
            # Split text
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text)
            
            # Create vector store
            embeddings = load_embedding_model()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            
            return vectorstore.as_retriever()
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
        finally:
            # Clean up temp file
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

# Main chat interface
if uploaded_file:
    retriever = process_pdf(uploaded_file)
    
    if retriever:
        # Initialize QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=load_llm(),
            retriever=retriever,
            return_source_documents=False
        )
        
        # Display chat history
        st.subheader("Chat")
        for qa in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(qa["question"])
            with st.chat_message("assistant"):
                st.write(qa["answer"])
        
        # Question input
        question = st.chat_input("Ask a question about the PDF...")
        
        if question:
            # Add user question to chat
            with st.chat_message("user"):
                st.write(question)
            
            # Get answer
            with st.spinner("Thinking..."):
                response = qa_chain({
                    "question": question, 
                    "chat_history": [(qa["question"], qa["answer"]) for qa in st.session_state.chat_history]
                })
                answer = response['answer']
            
            # Add assistant response to chat
            with st.chat_message("assistant"):
                st.write(answer)
            
            # Update chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer
            })
else:
    st.info("Please upload a PDF file to get started")