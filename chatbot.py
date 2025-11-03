# rag_ollama.py
from glob import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

# ------------------------------
# 1) Load documents
# ------------------------------

docs = []
for path in glob(r"C:\Users\Windows\Downloads\ensemble_learning_summary.pdf", recursive=True):
    docs.extend(PyPDFLoader(path).load())

# ------------------------------
# 2) Split into chunks
# ------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

print(f"Loaded {len(chunks)} chunks")

# ------------------------------
# 3) Embeddings + ChromaDB
# ------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # stores locally
)

# ------------------------------
# 4) LLaMA 3 via Ollama
# ------------------------------
llm = ChatOllama(model="smollm2")   # make sure you pulled llama3 using `ollama pull llama3`

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever()
)

# ------------------------------
# 5) Ask a question
# ------------------------------
question = ("what is Generative model?"
            "then what is Fine tuning in Generative AI"

            )
answer = qa.invoke({"query":question})
print(answer)
print("\n==============================")
print("Q:", question)
print("A:", answer)
print("==============================")

