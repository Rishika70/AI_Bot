import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.vectorstores import Milvus
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import WebLoader
import re

# Set the OpenAI API key securely
OPENAI_API_KEY = "sk-proj-d5NUtCGYR7lx6hC3tBGeT3BlbkFJFzjB0hCJeKMCmB251JC0"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load the LLM
llm = OpenAI(model="text-davinci-003", api_key=OPENAI_API_KEY)

# Initialize the vector store
vector_store = Milvus("cuda_docs", "HNSW", connection_args={"uri": "./cuda_docs.db"})

# Create a text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Load the CUDA documentation
loader = WebLoader("https://docs.nvidia.com/cuda/")
documents = loader.load()
docs = text_splitter.split_documents(documents)

# Create the vector database with embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vector_db = Milvus.from_documents(docs, embeddings, connection_args={"uri": "./cuda_docs.db"})

# Define a function to retrieve documents
def retrieve(query):
    try:
        # Remove punctuation and convert to lowercase
        query = re.sub(r'[^\w\s]', '', query).lower()
        
        # Retrieve documents using the vector store
        results = vector_db.similarity_search(query, k=10)
        return results
    except Exception as e:
        st.error(f"Error: {e}")
        return []

# Create a Streamlit app
st.title("CUDA Question Answering")
st.write("Ask a question about CUDA, GPU acceleration, or parallel computing:")

query = st.text_input("Question")

if st.button("Ask"):
    results = retrieve(query)
    st.write("Results:")
    for result in results:
        st.write(f"**{result.metadata['title']}**")
        st.write(result.text[:200] + "...")
        st.write("---")
