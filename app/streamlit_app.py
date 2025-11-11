import streamlit as st
import os
import traceback

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

from RagFullPipeline import rag_advanced, initialize_llm, RagRetriever, EmbeddingManager, VectorStore

st.set_page_config(
    page_title="Law Study Buddy",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .response-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 10px 0;
    }
    .source-item {
        background-color: #e9ecef;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚖️ Law Study Buddy</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Legal Research Assistant")

@st.cache_resource
def load_rag_components():
    try:
        st.sidebar.write("✅ Imports successful")

        embedding_manager = EmbeddingManager()
        st.sidebar.write("✅ Embedding manager loaded")

        vectorstore = VectorStore()
        st.sidebar.write("✅ Vector store loaded")

        llm = initialize_llm()
        if llm:
            st.sidebar.write("✅ LLM initialized")
        else:
            st.sidebar.error("❌ LLM failed - Check GROQ_API_KEY")
            return None, None

        retriever = RagRetriever(vectorstore, embedding_manager)
        st.sidebar.write("✅ Retriever ready")

        return retriever, llm

    except Exception as e:
        st.sidebar.error(f"❌ RAG initialization failed: {str(e)}")
        st.sidebar.code(traceback.format_exc())
        return None, None

with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=10,
        value=5,
        help="How many legal documents to retrieve"
    )

    min_score = st.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Minimum similarity score for sources"
    )

    return_context = st.checkbox(
        "Show full context",
        value=False,
        help="Display the full text of retrieved documents"
    )

    st.header("📊 System Status")
    retriever, llm = load_rag_components()

    if retriever and llm:
        st.success("✅ RAG System: Ready")
        try:
            doc_count = retriever.vector_store.collection.count()
            st.info(f"📚 Documents: {doc_count}")
        except:
            st.info("📚 Documents: Loading...")
    else:
        st.error("❌ RAG System: Not Ready")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Ask a Legal Question")

    query = st.text_area(
        "Enter your legal question:",
        placeholder="E.g., What are the elements required to prove negligence in tort law?",
        height=100
    )

    with st.expander("Advanced Options"):
        custom_prompt = st.text_area(
            "Custom instructions (optional):",
            placeholder="E.g., Focus on Nigerian case law and provide practical examples...",
            height=80
        )

    if st.button("🚀 Get Legal Answer", type="primary", use_container_width=True):
        if not query.strip():
            st.error("Please enter a legal question")
        elif not retriever or not llm:
            st.error("RAG system not initialized. Please check system status.")
        else:
            try:
                with st.spinner("🔍 Researching legal sources..."):
                    final_query = query
                    if custom_prompt:
                        final_query = f"{query}\n\nAdditional instructions: {custom_prompt}"

                    result = rag_advanced(
                        query=final_query,
                        retriever=retriever,
                        llm=llm,
                        top_k=top_k,
                        min_score=min_score,
                        return_context=return_context
                    )

                    st.markdown("### 📝 Legal Analysis")
                    st.markdown('<div class="response-box">', unsafe_allow_html=True)
                    st.write(result["answer"])
                    st.markdown('</div>', unsafe_allow_html=True)

                    confidence = result.get("confidence", 0)
                    st.metric("Confidence Score", f"{confidence:.2%}")

                    st.subheader("📚 Legal Sources")
                    sources = result.get("sources", [])

                    if sources:
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"Source {i}: {source['source']} (Score: {source['score']:.2f})"):
                                st.write(f"**Preview:** {source['preview']}")
                                st.write(f"**Page:** {source.get('page', 'N/A')}")
                                st.write(f"**Relevance Score:** {source['score']:.3f}")
                    else:
                        st.info("No specific sources retrieved for this query")

                    if return_context and result.get("context"):
                        st.subheader("📖 Full Context")
                        st.text_area("Retrieved Context", result["context"], height=200)

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.code(traceback.format_exc())

with col2:
    st.subheader("💡 Example Questions")

    examples = [
        "What constitutes murder under Nigerian criminal law?",
        "Explain the requirements for a valid contract",
        "What are the defenses to defamation?",
        "How does the statute of limitations work in tort cases?",
        "What is the difference between theft and robbery?",
        "Explain the concept of mens rea in criminal law"
    ]

    for example in examples:
        if st.button(example, key=example, use_container_width=True):
            st.session_state.last_query = example
            st.rerun()

    st.markdown("---")
    st.subheader("⚡ Quick Actions")

    if st.button("🔄 Reload RAG System", use_container_width=True):
        load_rag_components.clear()
        st.rerun()

st.markdown("---")
st.markdown(
    "⚠️ **Disclaimer:** This AI assistant provides legal information for educational purposes only. "
    "It does not constitute legal advice. Always consult a qualified attorney for legal matters."
)