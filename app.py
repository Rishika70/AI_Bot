import os
import streamlit as st
from langchain.llm import LLM
from langchain_milvus.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.document_loaders import WebLoader

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-proj-d5NUtCGYR7lx6hC3tBGeT3BlbkFJFzjB0hCJeKMCmB251JC0"

# Load the LLM
llm = LLM("deepset/roberta-base-squad2")

# Load the vector store
vector_store = Milvus("cuda_docs", "HNSW")

# Create a text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Load the CUDA documentation
loader = WebLoader("https://docs.nvidia.com/cuda/")
documents = loader.load()
docs = text_splitter.split_documents(documents)

# Create the vector database
embeddings = OpenAIEmbeddings()
vector_db = Milvus.from_documents(docs, embeddings, connection_args={"uri": "./cuda_docs.db"})

# Define a function to retrieve documents
def retrieve(query):
    # Retrieve documents using the vector store
    results = vector_store.query(query, k=10)
    return results

# Create a Streamlit app
st.title("CUDA Question Answering")
st.write("Ask a question about CUDA, GPU acceleration, or parallel computing:")

query = st.text_input("Question")

if st.button("Ask"):
    results = retrieve(query)
    st.write("Results:")
    for result in results:
        st.write(result)