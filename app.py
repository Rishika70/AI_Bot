from typing import Any, Dict
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.embeddings import Embeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.embeddings import Embeddings


def load_and_split_documents(urls: list[str]) -> list[Document]:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    return text_splitter.split_documents(docs_list)


def add_documents_to_milvus(
    doc_splits: list[Document], embedding_model: Embeddings, connection_args: Any
):
    vectorstore = Milvus.from_documents(
        documents=doc_splits,
        collection_name="rag_milvus",
        embedding=embedding_model,
        connection_args=connection_args,
    )
    return vectorstore.as_retriever()


# Initialize the components
urls = [
    "https://docs.nvidia.com/cuda/",
]

doc_splits = load_and_split_documents(urls)
embedding_model = HuggingFaceEmbeddings()
connection_args = {"uri": "./milvus_rag.db"}
retriever = add_documents_to_milvus(doc_splits, embedding_model, connection_args)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

retrieval_grader_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Here is the retrieved document: 
    {document}
    Here is the user question: 
    {question}""",
    input_variables=["question", "document"],
)

answer_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Context: {context} 
    Answer:""",
    input_variables=["question", "context"],
)

hallucination_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Here are the facts:
    {documents} 
    Here is the answer: 
    {generation}""",
    input_variables=["generation", "documents"],
)

question_router_prompt = PromptTemplate(
    template="""You are an expert at routing a user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and no preamble or explanation. 
    Question to route: 
    {question}""",
    input_variables=["question"],
)

# Create a Streamlit app
st.title("CUDA Question Answering")
st.write("Ask a question about CUDA, GPU acceleration, or parallel computing:")

query = st.text_input("Question")

if st.button("Ask"):
    results = (query)
    st.write("Results:")
    for result in results:
        st.write(f"**{result.metadata.get('title', 'No Title')}**")
        st.write(result.text[:200] + "...")
        st.write("---")

if __name__ == "__main__":
    st.info("Loading CUDA documentation and initializing vector database. Please wait...")
