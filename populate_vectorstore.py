from RagFullPipeline import process_all_pdfs, split_documents, EmbeddingManager, VectorStore

# 1️⃣ Load PDFs
documents = process_all_pdfs(r"C:\Users\DELL\LawSchool-CLEAN\app")


# 2️⃣ Split into chunks
split_docs = split_documents(documents)

# 3️⃣ Generate embeddings
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in split_docs])

# 4️⃣ Add to vector store
vectorstore = VectorStore()
vectorstore.add_documents(split_docs, embeddings)

print(f"Vector store now contains {vectorstore.collection.count()} documents")
