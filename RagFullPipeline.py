import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import uuid
import hashlib
import shutil

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()


def get_groq_api_key():
    if 'GROQ_API_KEY' in os.environ:
        return os.environ['GROQ_API_KEY']
    try:
        from dotenv import load_dotenv
        load_dotenv()
        if 'GROQ_API_KEY' in os.environ:
            return os.environ['GROQ_API_KEY']
    except:
        pass
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            return st.secrets['GROQ_API_KEY']
    except:
        pass
    return None


GROQ_API_KEY = get_groq_api_key()


def process_all_pdfs(pdf_directory):
    all_documents = []
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    for pdf in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf))
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_file'] = pdf.name
                doc.metadata['file_type'] = 'pdf'
            all_documents.extend(documents)
        except:
            continue
    return all_documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True)


class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents",
                 persist_directory: str = "./prebuilt_vector_store",
                 use_persistent: bool = True,
                 pdf_folder: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_persistent = use_persistent
        self.pdf_folder = pdf_folder

        # Initialize ChromaDB client with error handling
        self.client = self._initialize_chroma_client()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"Description": "PDF document embedding for RAG"}
        )

        # Populate with documents if collection is empty and PDF folder is provided
        if self.collection.count() == 0 and self.pdf_folder:
            print(f"Populating vector store from PDF folder: {self.pdf_folder}")
            self.populate_from_pdfs(self.pdf_folder)
        else:
            print(f"Vector store loaded with {self.collection.count()} documents")

    def _initialize_chroma_client(self):
        """Initialize ChromaDB client with proper error handling"""
        import os
        import shutil

        if not self.use_persistent:
            return chromadb.EphemeralClient()

        # For persistent storage, handle settings conflicts
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Try without settings first (most compatible)
                client = chromadb.PersistentClient(path=self.persist_directory)
                return client
            except ValueError as e:
                if "already exists" in str(e) and attempt < max_retries - 1:
                    # Delete the conflicting database and retry
                    print(f"Database conflict detected. Removing {self.persist_directory}...")
                    if os.path.exists(self.persist_directory):
                        shutil.rmtree(self.persist_directory)
                    os.makedirs(self.persist_directory, exist_ok=True)
                else:
                    # Fallback to ephemeral on final failure
                    print(f"Persistent storage failed: {e}. Using ephemeral client.")
                    return chromadb.EphemeralClient()

        # Final fallback
        return chromadb.EphemeralClient()

    def _doc_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        existing_ids = set()
        try:
            existing_docs = self.collection.get()
            if existing_docs and "metadatas" in existing_docs:
                for meta in existing_docs["metadatas"]:
                    for m in meta:
                        if "doc_hash" in m:
                            existing_ids.add(m["doc_hash"])
        except:
            pass
        new_ids, new_texts, new_metas, new_embeddings = [], [], [], []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            text_hash = self._doc_hash(doc.page_content)
            if text_hash in existing_ids:
                continue
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            new_ids.append(doc_id)
            new_texts.append(doc.page_content)
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadata["doc_hash"] = text_hash
            new_metas.append(metadata)
            new_embeddings.append(embedding.tolist())
        if new_ids:
            batch_size = 1000
            for i in range(0, len(new_ids), batch_size):
                end_idx = min(i + batch_size, len(new_ids))
                self.collection.add(
                    ids=new_ids[i:end_idx],
                    embeddings=new_embeddings[i:end_idx],
                    metadatas=new_metas[i:end_idx],
                    documents=new_texts[i:end_idx]
                )
            print(f"Added {len(new_ids)} new documents to vector store")

    def populate_from_pdfs(self, pdf_folder: str):
        """Load and process PDFs from the specified folder"""
        if not os.path.exists(pdf_folder):
            print(f"PDF folder not found: {pdf_folder}")
            return

        print(f"Processing PDFs from: {pdf_folder}")
        documents = process_all_pdfs(pdf_folder)

        if not documents:
            print("No PDF documents found or processed")
            return

        print(f"Loaded {len(documents)} documents from PDFs")
        split_docs = split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")

        embedding_manager = EmbeddingManager()
        print("Generating embeddings...")
        embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in split_docs])
        print(f"Generated {len(embeddings)} embeddings")

        self.add_documents(split_docs, embeddings)
        print("PDF population completed")


class RagRetriever:
    def __init__(self, vector_store, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            retrieved_results = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results["ids"][0]
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_results.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i
                        })
            return retrieved_results
        except:
            return []


def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    if not GROQ_API_KEY:
        return {'answer': 'GROQ_API_KEY not found.', 'sources': [], 'confidence': 0.0, 'context': ''}
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}
    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{'source': doc['metadata'].get('source_file', 'unknown'),
                'page': str(doc['metadata'].get('page', 'unknown')),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'} for doc in results]
    confidence = max([doc['similarity_score'] for doc in results])
    prompt = f"""
You are an expert AI legal scholar.
Context:
{context if context else "No context found."}
Question:
{query}
Answer:
"""
    response = llm.invoke(prompt)
    output = {'answer': response.content, 'sources': sources, 'confidence': confidence}
    if return_context:
        output['context'] = context
    return output


def initialize_llm():
    if not GROQ_API_KEY:
        return None
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.1, max_tokens=3035)