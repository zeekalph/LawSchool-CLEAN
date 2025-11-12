import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import uuid
import hashlib

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

    print(f"found {len(pdf_files)} pdf_files to process")

    for pdf in pdf_files:
        print(f"\nProcessing: {pdf.name}")
        try:
            loader = PyMuPDFLoader(str(pdf))
            documents = loader.load()

            for doc in documents:
                doc.metadata['source_file'] = pdf.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")

        except Exception as e:
            print(f"Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print(f"\nExample chunk:")
        print(f"content: {split_docs[0].page_content[:200]}...")
        print(f"metadata: {split_docs[0].metadata}")

    return split_docs


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings


class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents",
                 persist_directory: str = "./prebuilt_vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._init_vectorstore()

    def _init_vectorstore(self):
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"Description": "PDF document embedding for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store {e}")
            raise

    def _doc_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        existing_ids = set()
        try:
            existing_docs = self.collection.get()
            if existing_docs and "metadatas" in existing_docs:
                for meta in existing_docs["metadatas"]:
                    for m in meta:
                        if "doc_hash" in m:
                            existing_ids.add(m["doc_hash"])
        except Exception:
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
            try:
                batch_size = 1000
                for i in range(0, len(new_ids), batch_size):
                    end_idx = min(i + batch_size, len(new_ids))
                    batch_ids = new_ids[i:end_idx]
                    batch_texts = new_texts[i:end_idx]
                    batch_metas = new_metas[i:end_idx]
                    batch_embeddings = new_embeddings[i:end_idx]

                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metas,
                        documents=batch_texts
                    )
                    print(f"Added batch {i // batch_size + 1}: {len(batch_ids)} documents")

                print(f"Added {len(new_ids)} new documents to the vector store")
                print(f"Total documents in collection: {self.collection.count()}")
            except Exception as e:
                print(f"Error adding documents to vectorstore: {e}")
                raise
        else:
            print("No new documents to add — skipping.")


class RagRetriever:
    def __init__(self, vector_store, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top_k: {top_k}, score_threshold: {score_threshold}")

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

                print(f"Retrieved {len(retrieved_results)} documents (after filtering)")
            else:
                print("No documents found")

            return retrieved_results

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    if not GROQ_API_KEY:
        return {
            'answer': 'GROQ_API_KEY not found. Please set it in Streamlit secrets or environment variables.',
            'sources': [], 'confidence': 0.0, 'context': ''}

    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {
            'answer': 'No relevant context found, it is either that your question is beyond the scope of this system or your prompt needs to be reconstructed.',
            'sources': [], 'confidence': 0.0, 'context': ''}

    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': str(doc['metadata'].get('page', 'unknown')),
        'score': doc['similarity_score'],
        'preview': doc['content'][:120] + '...'
    } for doc in results]

    confidence = max([doc['similarity_score'] for doc in results])

    prompt = f"""
You are an expert AI legal scholar with deep knowledge of Nigerian and general common law.
You have access to authoritative sources (textbooks, statutes, case law).

Your task:
- Answer the question thoroughly as if teaching a law student.
- Cite relevant cases with full names and years.
- Include key quotes from textbooks, statutes, and judgments when available.
- Mention obiter dicta where relevant.
- Provide examples to illustrate concepts clearly.
- Reference statutes and explain how they interact with common law.
- Analyze real-world situations, hypothetical scenarios and legal problems, applying the law to solve them step by step.
- When presented with an hypothetical or a scenario based question by the user, analyze the situation, pin point the issue and then solve it
- Give a reasoned legal opinion for practical scenarios, highlighting potential outcomes.
- Include a clear caveat where applicable (e.g., "this is not legal advice").
- Organize your answer with headings, subheadings, and bullet points for clarity.
- Ensure your answer is accurate, academically rigorous, and detailed.
- When possible, interpret complex legal concepts in the context of everyday real-world cases.

Context (from indexed textbooks, statutes, and case materials):
{context if context else "No context found in vector store; answer using your general legal knowledge and reasoning."}

Question:
{query}

Answer:
"""
    response = llm.invoke(prompt)

    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence,
    }

    if return_context:
        output['context'] = context

    return output


def rag_simple(query, retriever, llm, top_k=5):
    if not GROQ_API_KEY:
        return "GROQ_API_KEY not found. Please set it in Streamlit secrets or environment variables."

    results = retriever.retrieve(query, top_k=top_k)

    context = "\n\n".join([doc['content'] for doc in results]) if results else ""

    if not context:
        return "No relevant context found for this query, it is either that the question if beyond the scope of this system or your prompt needs to be reconstructed."

    prompt = f"""You are an AI that is greatly knowledgeable about the law.
        Use the following context to answer the question concisely

        Context:
        {context}

        Question:
        {query}

        Answer:"""

    response = llm.invoke(prompt)
    return response.content


def initialize_llm():
    if not GROQ_API_KEY:
        return None

    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=3035
    )