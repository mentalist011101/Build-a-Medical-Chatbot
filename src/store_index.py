from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone 
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

extrated_data = load_pdf_files('/content/drive/MyDrive/data_medical')
minimal_docs = filter_to_minimal_docs(extrated_data)
texts_chunks = text_split(minimal_docs)

embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY


pc = Pinecone(api_key=pinecone_api_key,)

index_name = "medicalchatbot"

if not pc.has_index(index_name):
  pc.create_index(
      name=index_name,
      dimension=384,
      metric="cosine", #Cosine
      spec = ServerlessSpec(cloud = "aws", region="us-east-1")
  )
index = pc.Index(index_name)

# Ensure the API key is set as an environment variable for langchain-pinecone
PINECONE_API_KEY = PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    embedding=embeddings,
    index_name=index_name
)