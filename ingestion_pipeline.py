from json import load
import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader,PyPDFLoader,UnstructuredImageLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def Load_multimodal_documents(docs_path="docs"):
    """
    Manually routes files to their appropriate loaders based on extension.
    """
    all_docs = []
    
    if not os.path.exists(docs_path):
        print(f"Error: Directory '{docs_path}' not found.")
        return []

    print(f"Scanning directory: {docs_path}...")
    
    for filename in os.listdir(docs_path):
        file_path = os.path.join(docs_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path)
            elif ext in [".jpg", ".jpeg", ".png"]:
                # Note: Requires 'tesseract' installed on your system for OCR
                loader = UnstructuredImageLoader(file_path)
            else:
                continue # Skip unsupported types
                
            all_docs.extend(loader.load())
            print(f" Loaded: {filename}")
            
        except Exception as e:
            print(f" Failed to load {filename}: {str(e)}")

    return all_docs

def main():
    raw_docs= Load_multimodal_documents(docs_path="docs")
    print("Starting the ingestion pipeline...")
    # Load documents 
       
    # chunking the files into smaller pieces to fit into the context window of the language model
    # Embedding and Storing the chunks in a vector database (Chroma) 
if __name__ == "__main__":
    main()