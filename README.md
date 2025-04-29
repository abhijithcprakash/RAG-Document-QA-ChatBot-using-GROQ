---

# 📄 Document Q&A Assistant 

An intelligent, PDF-powered Q&A chatbot built using **Streamlit**, **LangChain**, **Groq API (Llama3)**, and **FAISS** for vector search.  
Users can upload any PDF, embed its content, and ask questions — with answers generated using **RAG (Retrieval-Augmented Generation)**.

---

## ⚙️ Project Overview

### 🧠 How It Works

1. **PDF Upload**  
   User uploads a PDF document through the Streamlit interface.

2. **Text Extraction & Splitting**  
   Text is extracted using `PyPDFLoader`, and split into manageable chunks using `RecursiveCharacterTextSplitter`.

3. **Vector Embedding**  
   The text chunks are converted into embeddings using `OllamaEmbeddings` with the `gemma:2b` model and stored in **FAISS**, a fast vector search library.

4. **RAG-based Question Answering**  
   When the user asks a question:
   - The retriever finds relevant content chunks from the embedded document.
   - These are passed as context to a **Groq-hosted Llama3 model** to generate a smart answer.

5. **Frontend UI**  
   The Streamlit app provides a clean interface with real-time feedback and styled components for user-friendly interaction.

---

## 🛠 Tech Stack

- **Streamlit** – UI and interactivity
- **LangChain** – For chaining prompt logic and RAG
- **Groq API** – LLM inference (Llama3-8b-8192)
- **OllamaEmbeddings** – For document vectorization
- **FAISS** – Vector similarity search
- **dotenv** – Managing environment variables

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/document-qa-assistant.git
cd document-qa-assistant
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Create a `.env` File
Add your API keys in a `.env` file at the project root:

```
GROQ_API_KEY=your-groq-api-key
LANGCHAIN_API_KEY=your-langchain-api-key
HUGGINGFACE_API=your-huggingface-api-key
```
> ✅ **GROQ_API_KEY** is used to access Llama3 from Groq.

---

## 🚀 Running the App

```bash
streamlit run app.py
```

> Replace `app.py` with your actual script name if different.

---

## 💡 Features

- 📤 Upload any PDF
- ⚡ Smart document embedding with `gemma:2b`
- 🤖 Intelligent responses using Llama3-8b via Groq API
- 🔍 Retrieval-Augmented Generation (RAG)
- 🎨 Beautiful, interactive UI
- 📚 View the context retrieved from your document

---

## 🔐 API Key Handling

- All API keys are securely loaded from `.env` using `python-dotenv`.
- No key is hardcoded in the script.
- Only the **Groq API Key** is used at runtime for LLM inference.

---


---

## 🧪 Demo Workflow

1. Upload a PDF file.
2. Click **"📚 Embed Document"**.
3. Type your question in the input box.
4. Get your answer — backed by relevant context from the document.
---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
