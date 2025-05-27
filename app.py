import json
import sys
import os
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
# data ingenstion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# vector embeddings and vector store
from langchain_community.vectorstores import FAISS

## LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.environ.get('AWS_REGION', 'us-west-1'),
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)
model_id = "amazon.titan-embed-image-v1"
bedrock_embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id=model_id
)

# data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data", glob="**/*.pdf")
    documents = loader.load()
    test_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    docs = test_splitter.split_documents(documents)
    return docs

# vector embedding and vector store.
def vector_store(docs):
    vector_store = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store.save_local("faiss_index")
    # return vector_store

def get_claude_llm():
    llm = Bedrock(
        client=bedrock,
        model_id='anthropic.claude-2',
        model_kwargs={
            "max_tokens": 2000,
        }
    )
    return llm

def get_llama3_llm():
    llm = Bedrock(
        client=bedrock,
        model_id="meta.llama3-70b-instruct-v1:0",
        model_kwargs={
            "max_gen_len": 2000,
            "temperature": 0.1,
            "top_p": 0.9
        }
    )
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know. Don't try to make up an answer.
<context>
{context}
</context>

Question: {question}
Assistant:
"""

prompt = PromptTemplate.from_template(prompt_template)

def get_response_llm(llm, vector_store, query):
    qa =  RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )
    
    response = qa.invoke({"query": query})
    return response['result'], response['source_documents']


def main():
    st.set_page_config(page_title="LLM App", page_icon=":robot_face:")
    st.title("LLM App with Amazon Bedrock")
    st.header("Chat with pdf using AWS Bedrock")
    
    user_question = st.text_area(
        "Ask a question about the PDF documents you uploaded",
        placeholder="Type your question here...",
        height=100
    )
    
    with st.sidebar:
        st.title("Update or create Vector Store")
        
        if st.button("Update Vector Store"):
            with st.status("Updating vector store..."):
                docs = data_ingestion()
                vector_store(docs)
            st.success("Vector store updated successfully!")
            
    if st.button("Claude Output"):
        with st.spinner("Generating response with Claude..."):
            llm = get_claude_llm()
            vectorstore = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings,
                allow_dangerous_deserialization=True
                )
            response, source_docs = get_response_llm(llm, vectorstore, user_question)
            
            st.subheader("Claude's Response")
            st.write(response)
            st.subheader("Source Documents")
            for doc in source_docs:
                st.write(doc.page_content)
    
    if st.button("Llama3 Output"):
        with st.spinner("Generating response with LLama3..."):
            llm = get_llama3_llm()
            vectorstore = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings,
                allow_dangerous_deserialization=True
                )
            response, source_docs = get_response_llm(llm, vectorstore, user_question)
            
            st.subheader("Llama3's Response: ")
            st.write(response)
            st.subheader("Source Documents")
            for doc in source_docs:
                st.write(doc.page_content)

if __name__ == "__main__":
    main()
    # sys.exit(main())