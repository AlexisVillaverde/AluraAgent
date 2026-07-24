import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def crear_base_conocimiento():
    print("1. Cargando documentos desde la carpeta 'data/'...")
    loader = PyPDFDirectoryLoader("data/")
    documentos = loader.load()
    
    if not documentos:
        print("Error: No se encontraron PDFs en la carpeta 'data/'.")
        return

    print(f"-> Se cargaron {len(documentos)} páginas.")

    print("2. Dividiendo el texto en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    fragmentos = text_splitter.split_documents(documentos)
    print(f"-> Se generaron {len(fragmentos)} fragmentos de texto.")

    print("3. Generando embeddings y guardando en ChromaDB...")
    # Usamos un modelo open-source ligero y rápido para los embeddings
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    
    print("¡Base de conocimiento creada exitosamente en la carpeta 'chroma_db'!")

if __name__ == "__main__":
    crear_base_conocimiento()