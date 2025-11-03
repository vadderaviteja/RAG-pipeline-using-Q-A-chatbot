# 📚 RAG Pipeline using Ollama + LangChain + ChromaDB

This project implements a Retrieval-Augmented Generation (RAG) system that uses a local LLM (smollm2 via Ollama) to answer questions based on a given PDF. Instead of relying on internet knowledge, the model retrieves context from your documents and generates accurate responses.

# 🚀 Features

Load & process PDF documents

Split text into chunks for better context handling

Generate embeddings using Sentence-Transformers

Store vectors in ChromaDB

Retrieve relevant text for each query

Answer questions using a local LLM (Ollama)

# 🧠 Tech Stack
Component	Tool
LLM	Ollama (smollm2)
Framework	LangChain
Embeddings	Sentence-Transformers (all-MiniLM-L6-v2)
Vector DB	ChromaDB
Language	Python
# 📂 Project Flow
PDF → Text Split → Embeddings → ChromaDB → Retriever → Ollama LLM → Answer

# ▶️ Run Instructions
pip install langchain langchain-community langchain-ollama chromadb sentence-transformers pypdf
ollama run smollm2
python rag_ollama.py

🎯 Output

Ask questions like:

What is a Generative Model?
What is fine-tuning in Generative AI?

The system will answer using only the PDF content — ensuring accuracy and no hallucination.

# ✅ Use Cases

Research document Q&A

Private company knowledge assistant

Offline AI assistant

Study material chatbot

📌 Future Enhancements

Streamlit UI

Support multiple PDFs

Upgrade model to Llama-3

QLoRA fine-tuning for domain data
