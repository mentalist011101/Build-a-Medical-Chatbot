from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import torch

# extraire le texte du pdf
def load_pdf_files(directory_path):
    loader = DirectoryLoader(
        directory_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
        )
    documents = loader.load()
    return documents



# filter le contenu des documents
def filter_to_minimal_docs(documents: List[Document]) -> List[Document]:
    """
    Given a list of document objects, return a new list of Document objects
    containing only 'source' in metadata and the 'page_content' of the original documents.
    """
    minimal_docs = []
    for doc in documents:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
             page_content=doc.page_content,
             metadata={'source':src}
             )
         )
    return minimal_docs

# Decouper les documents en chuncks
def text_split(documents):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,
      chunk_overlap=20,
      #length_function=len,
      #add_start_index=True,
  )
  texts_chunks = text_splitter.split_documents(documents)
  return texts_chunks



# telecharger l'embedding de HuggingFace
def download_embeddings():
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
  embeddings = HuggingFaceEmbeddings(
      model_name= model_name,
      model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
      )
  return embeddings