import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import time
from huggingface_hub import login

###
groq_api_key = st.secrets["api_keys"]["GROQ_API_KEY"]
langchain_api_key = st.secrets["api_keys"]["LANGCHAIN_API_KEY"]
huggingface_api_key = st.secrets["api_keys"]["HUGGINGFACE_API"]

login(huggingface_api_key)  # This will authenticate Hugging Face

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>

    Question: {input}
    """
)

# Function to create vector embeddings
def create_vector_embedding(file_path):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if len(docs) == 0:
            st.error("No content found in the uploaded PDF file.")
            return
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        final_documents = text_splitter.split_documents(docs)
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.success("✅ Document successfully embedded!")

# Streamlit Page Configuration
st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    body {
        background-color: #F0F2F6;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2E4053;
    }
    .subheader {
        font-size: 20px;
        color: #566573;
    }
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("# 📄 Document Q&A Assistant")
st.markdown("###### Upload any PDF document and ask questions intelligently using Groq and Llama3 🚀")

# Upload Document Section
st.markdown("---")
st.subheader("📤 Upload Your Document (PDF)")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Query Input
st.markdown("---")
st.subheader("🔍 Ask Questions About Your Document")
user_prompt = st.text_input("Enter your question:")

# Document Embedding
if uploaded_file is not None:
    with open("temp_document.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.button("📚 Embed Document"):
        with st.spinner("🔄 Embedding your document, please wait..."):
            create_vector_embedding("temp_document.pdf")

# Query Answering
if user_prompt:
    if "vectors" not in st.session_state:
        st.error("⚠️ Please upload and embed your document first!")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("🤔 Thinking... generating a smart answer..."):
            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            elapsed = time.process_time() - start
        
        # Show response
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 AI Assistant's Answer")
        st.write(response['answer'])
        st.markdown('</div>', unsafe_allow_html=True)

        # Optional: Show retrieved documents
        with st.expander("🔎 See Retrieved Document Content"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.markdown("---")
else:
    st.info("💬 Please ask a question after uploading your document.")

