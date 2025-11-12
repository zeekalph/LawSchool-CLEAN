import os
import sys
import time
sys.path.append('app')

from app.RagFullPipeline import process_all_pdfs, split_documents, EmbeddingManager, VectorStore
import shutil

print("Building vector store...")
documents = process_all_pdfs("app")
split_docs = split_documents(documents)
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in split_docs])

vector_store = VectorStore()
vector_store.add_documents(split_docs, embeddings)

print(f"Vector store created with {vector_store.collection.count()} documents")

# Close the vector store first
del vector_store
time.sleep(2)

# Copy the vector store to prebuilt location
if os.path.exists("./prebuilt_vector_store"):
    shutil.rmtree("./prebuilt_vector_store")
shutil.copytree("./vector_store", "./prebuilt_vector_store")
print("Pre-built vector store created!")