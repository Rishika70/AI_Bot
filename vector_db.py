
from milvus import Milvus
from transformers import AutoTokenizer, AutoModel
import numpy as np

def create_vector_db(chunks):
    milvus_client = Milvus()

    # Create a collection
    collection_name = "cuda_docs"
    milvus_client.create_collection(collection_name, field_params=[
        {"name": "text", "type": "VarChar", "max_length": 1024 * 1024},
        {"name": "embedding", "type": "FloatVector", "dim": 768}
    ])

    # Insert data
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())

    # Insert data into the collection
    milvus_client.insert(collection_name, [{"text": chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)])

    # Create an index
    milvus_client.create_index(collection_name, "embedding")

def chunk_data(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    # Load the text from the webpage
    import requests
    from bs4 import BeautifulSoup

    url = "https://docs.nvidia.com/cuda/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()

    chunks = chunk_data(text)
    create_vector_db(chunks)