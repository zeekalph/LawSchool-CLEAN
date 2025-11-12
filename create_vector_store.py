import os
import sys
import time
from RagFullPipeline import process_all_pdfs, split_documents, EmbeddingManager, VectorStore
import shutil

PDF_DIR = "app"  # folder where your PDFs are
PREBUILT_DIR = "./prebuilt_vector_store"

# Remove old prebuilt store if exists
if os.path.exists(PREBUILT_DIR):
    shutil.rmtree(PREBUILT_DIR)

print("Building vector store...")

# 1. Load documents
documents = process_all_pdfs(PDF_DIR)
if not documents:
    print("No documents found in", PDF_DIR)
    sys.exit(1)

# 2. Split documents into chunks
split_docs = split_documents(documents)

# 3. Generate embeddings
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in split_docs])

# 4. Create vector store and add documents
vector_store = VectorStore(persist_directory=PREBUILT_DIR)
vector_store.add_documents(split_docs, embeddings)

print(f"Vector store created with {vector_store.collection.count()} documents")

# Optional: cleanup
del vector_store
time.sleep(1)

print("Pre-built vector store ready at:", PREBUILT_DIR)
