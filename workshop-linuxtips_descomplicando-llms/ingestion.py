import os
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as pc
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_env = os.environ.get('PINECONE_ENV')
openai_apikey = os.environ.get('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

loader = PyPDFLoader("docs/CLT.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

pinecone = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = 'linuxtips'
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )

index = pinecone.Index(index_name)

docsearch = Pinecone.from_texts([t.page_content for i in texts], embeddings, index_name=index_name)

# Test
# query = "o que são férias?"
# docs = docsearch.similarity_search(query)
# print(docs[0])