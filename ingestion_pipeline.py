import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredImageLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import time # Added for potential rate limiting

# 1. Load and Verify Environment
load_dotenv()
github_token = os.getenv("GITHUB_TOKEN")

if not github_token:
    raise ValueError("CRITICAL ERROR: GITHUB_TOKEN not found. "
                     "Please ensure your .env file contains GITHUB_TOKEN=your_token_here")
else:
    print(f"Success: GitHub Token detected (Ends in: ...{github_token[-4:]})")


def Load_multimodal_documents(docs_path="docs"):
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
                loader = UnstructuredImageLoader(file_path)
            else:
                continue
            all_docs.extend(loader.load())
            print(f" Loaded: {filename}")
        except Exception as e:
            print(f" Failed to load {filename}: {str(e)}")
    return all_docs


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    # Upgraded to RecursiveCharacterTextSplitter for better semantic integrity
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks


def create_vector_store(chunks, persistent_directory="db/chroma_db"):
    """Creates a vector store by batching documents to avoid payload limits."""
    print("Initializing GitHub-hosted embeddings...")
    
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=github_token,
        openai_api_base="https://models.inference.ai.azure.com", 
        dimensions=1024
    )
    
    # 1. Initialize the Chroma client first
    print(f"Creating/Loading vector store at: {persistent_directory}")
    vector_store = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # 2. Define Batch Size (How many chunks to send at once)
    batch_size = 50 
    print(f"Adding {len(chunks)} chunks in batches of {batch_size}...")

    # 3. Loop through the chunks and add them incrementally
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f" Processing batch {i//batch_size + 1} ({len(batch)} chunks)...")
        
        try:
            vector_store.add_documents(batch)
            # Short pause to be kind to the GitHub API
            time.sleep(1) 
        except Exception as e:
            print(f" CRITICAL ERROR on batch {i//batch_size + 1}: {e}")
            break

    print("Finished creating vector store.")
    return vector_store


def main():
    print("Starting the GitHub Models Ingestion Pipeline...")
    raw_docs = Load_multimodal_documents(docs_path="docs")
    
    if not raw_docs:
        print("No documents found. Exiting.")
        return

    split_docs = split_documents(raw_docs)
    create_vector_store(chunks=split_docs)

if __name__ == "__main__":
    main()