import os
import chromadb
import uuid
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import ollama

# Paths
txt_dir = "./data/full_contract_txt"  # Adjust path if needed
persist_directory = "./VectorStore"

# Ensure persist directory exists
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=persist_directory)

# Create or retrieve collection
collection_name = "cuad_contracts"
try:
    collection = client.get_collection(collection_name)
except chromadb.errors.InvalidCollectionException:  # Adjusted to use the correct exception
    collection = client.create_collection(collection_name)


# Load contracts from text files
def load_contracts(txt_dir):
    """Load contracts as documents from the specified directory."""
    documents = []
    for file_name in os.listdir(txt_dir):
        if file_name.endswith(".txt"):
            with open(os.path.join(txt_dir, file_name), "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({"content": content, "metadata": {"file_name": file_name}})
    return documents


# Generate embeddings and add to ChromaDB
def process_contracts():
    """Process contracts and add their embeddings to ChromaDB."""
    documents = load_contracts(txt_dir)
    splitter = CharacterTextSplitter(chunk_size=3400, chunk_overlap=300)

    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        for idx, chunk in enumerate(chunks):
            # Generate embedding
            try:
                response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
                embedding = response.get("embedding")
                if embedding:
                    collection.add(
                        documents=[chunk],
                        metadatas=[{"source": doc["metadata"]["file_name"], "chunk_id": idx}],
                        ids=[str(uuid.uuid4())],
                        embeddings=[embedding]
                    )
            except Exception as e:
                print(f"Error processing chunk {idx} of {doc['metadata']['file_name']}: {e}")


if __name__ == "__main__":
    process_contracts()
    print("Contracts processed and stored in ChromaDB.")
