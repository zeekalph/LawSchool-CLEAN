import streamlit as st
import os
import traceback
import time

st.set_page_config(
    page_title="Law Study Buddy",
    page_icon="⚖️",
    layout="wide"
)

# Configure for faster startup
st.session_state.setdefault('rag_initialized', False)
st.session_state.setdefault('vector_store_ready', False)

st.sidebar.write("### Secrets Debug")
st.sidebar.write(f"Has secrets: {hasattr(st, 'secrets')}")

if hasattr(st, 'secrets'):
    st.sidebar.write(f"All secrets: {list(st.secrets.keys())}")
    if 'GROQ_API_KEY' in st.secrets:
        st.sidebar.write(f"API Key found: {st.secrets['GROQ_API_KEY'][:10]}...")
        os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
    else:
        st.sidebar.write("❌ GROQ_API_KEY not found in secrets")
else:
    from dotenv import load_dotenv

    load_dotenv()

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .response-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚖️ Law Study Buddy</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Legal Research Assistant")


def initialize_minimal_rag():
    """Initialize only essential components"""
    try:
        from RagFullPipeline import initialize_llm, EmbeddingManager

        # Initialize LLM first (fast)
        llm = initialize_llm()
        if not llm:
            st.error("❌ LLM failed - Check GROQ_API_KEY")
            return None, None, None

        # Initialize embedding manager (fast)
        embedding_manager = EmbeddingManager()

        return llm, embedding_manager, "Minimal components loaded"

    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return None, None, str(e)


def load_vector_store_lazy():
    """Load vector store only when needed"""
    try:
        from RagFullPipeline import VectorStore, RagRetriever

        # Check if prebuilt vector store exists
        persist_dir = "./prebuilt_vector_store"
        if not os.path.exists(persist_dir):
            st.warning("⚠️ No prebuilt vector store found. Using demo mode.")
            return None, "No vector store available"

        # Load existing vector store (should be fast)
        vectorstore = VectorStore(
            persist_directory=persist_dir,
            use_persistent=True,
            pdf_folder=None  # Don't populate on load
        )

        return vectorstore, f"Vector store loaded with {vectorstore.collection.count()} documents"

    except Exception as e:
        return None, f"Vector store error: {str(e)}"


# Initialize minimal components
with st.sidebar:
    st.header("📊 System Status")

    if not st.session_state.rag_initialized:
        with st.spinner("Loading AI components..."):
            llm, embedding_manager, status_msg = initialize_minimal_rag()
            if llm and embedding_manager:
                st.session_state.llm = llm
                st.session_state.embedding_manager = embedding_manager
                st.session_state.rag_initialized = True
                st.success("✅ AI Components Ready")
            else:
                st.error(f"❌ {status_msg}")

    if st.session_state.rag_initialized and not st.session_state.vector_store_ready:
        with st.spinner("Checking vector store..."):
            vectorstore, vector_status = load_vector_store_lazy()
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = RagRetriever(vectorstore, st.session_state.embedding_manager)
                st.session_state.vector_store_ready = True
                st.success("✅ Vector Store Ready")
                st.info(f"📚 Documents: {vectorstore.collection.count()}")
            else:
                st.warning("⚠️ " + vector_status)

with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=10,
        value=3,  # Reduced for performance
        help="How many legal documents to retrieve"
    )

    min_score = st.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.3,  # Increased for better quality
        step=0.1,
        help="Minimum similarity score for sources"
    )

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Ask a Legal Question")

    query = st.text_area(
        "Enter your legal question:",
        placeholder="E.g., What are the elements required to prove negligence in tort law?",
        height=100
    )

    if st.button("🚀 Get Legal Answer", type="primary", use_container_width=True):
        if not query.strip():
            st.error("Please enter a legal question")
        elif not st.session_state.get('rag_initialized'):
            st.error("AI components not ready. Please check system status.")
        else:
            try:
                with st.spinner("🔍 Researching legal sources..."):
                    from RagFullPipeline import rag_advanced

                    # Use retriever if available, otherwise use LLM only
                    if st.session_state.get('retriever'):
                        result = rag_advanced(
                            query=query,
                            retriever=st.session_state.retriever,
                            llm=st.session_state.llm,
                            top_k=top_k,
                            min_score=min_score,
                            return_context=False
                        )
                    else:
                        # Fallback to LLM-only response
                        response = st.session_state.llm.invoke(f"""
                        You are an expert legal scholar. Answer this legal question:

                        {query}

                        Provide a comprehensive legal analysis with relevant principles and examples.
                        """)
                        result = {
                            'answer': response.content,
                            'sources': [],
                            'confidence': 0.8
                        }

                    st.markdown("### 📝 Legal Analysis")
                    st.markdown('<div class="response-box">', unsafe_allow_html=True)
                    st.write(result["answer"])
                    st.markdown('</div>', unsafe_allow_html=True)

                    if result.get("confidence"):
                        st.metric("Confidence Score", f"{result['confidence']:.2%}")

                    if result.get("sources"):
                        st.subheader("📚 Legal Sources")
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"Source {i}: {source['source']} (Score: {source['score']:.2f})"):
                                st.write(f"**Preview:** {source['preview']}")
                                st.write(f"**Page:** {source.get('page', 'N/A')}")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

with col2:
    st.subheader("💡 Example Questions")

    examples = [
        "What constitutes murder under Nigerian criminal law?",
        "Explain the requirements for a valid contract",
        "What are the defenses to defamation?",
        "How does the statute of limitations work in tort cases?"
    ]

    for example in examples:
        if st.button(example, key=example, use_container_width=True):
            st.session_state.last_query = example
            st.rerun()

    st.markdown("---")
    st.subheader("⚡ Quick Actions")

    if st.button("🔄 Check Vector Store", use_container_width=True):
        st.session_state.vector_store_ready = False
        st.rerun()

st.markdown("---")
st.markdown(
    "⚠️ **Disclaimer:** This AI assistant provides legal information for educational purposes only. "
    "It does not constitute legal advice. Always consult a qualified attorney for legal matters."
)