"""
app.py

This Streamlit application sets up an interactive interface for the IISc Conversational Academic Navigator (I-CAN).
Users can test the RAG pipeline with sample chunks .

Features:
- Test mode with hardcoded document chunks.
- Interactive question input and retrieval-augmented responses.
- Sidebar displays current configuration parameters.
"""
import sys
import os

# Add /mount/src/i-can-iisc-conversational-academic-navigator/src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from typing import List, Tuple
import streamlit as st

# Import the function to initialize the RAG pipeline
from retrieval.rag_pipeline import initialize_rag_pipeline

# Import application configuration constants
from config.settings import MODEL_NAME, TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP

# LangChain document type for wrapping text chunks
from langchain.docstore.document import Document


def main() -> None:
    """
    Main entry point for the Streamlit app. Sets up UI elements, handles user interactions,
    and displays retrieval-augmented answers.
    """
    # Application title and subtitle
    st.title("I-CAN: IISc Conversational Academic Navigator")

    # Test mode section: Hardcode chunks for demonstration/testing
    if st.button("Run Test with Sample Data"):
        sample_texts: List[str] = [
            (
                "Streamlit is an open-source Python library that makes it easy to create and "
                "share beautiful, custom web apps for machine learning and data science. "
                "In just a few lines of code, you can build powerful user interfaces that "
                "can display charts, maps, tables, and interactive widgets."
            ),
            (
                "LangChain is a framework for developing applications powered by language models. "
                "It provides modular components for prompt management, memory, chains, agents, "
                "and retrieval, enabling developers to build complex, production-ready chains."  
            ),
            (
                "Retrieval-Augmented Generation (RAG) combines retrieval techniques with generative "
                "language models. By first retrieving relevant documents and then using a language "
                "model to generate responses conditioned on those documents, RAG systems improve "
                "accuracy and grounding of outputs."
            ),
            (
                "Configurable pipelines allow teams to adjust model parameters, chunking logic, "
                "and retrieval settings without changing the core code. Parameters like CHUNK_SIZE "
                "and CHUNK_OVERLAP help balance retrieval granularity and performance."
            ),
            (
                "Dr. Ratikanta Behera is an Assistant Professor (May 2022–present) in the Department "
                "of Computational and Data Sciences at the Indian Institute of Science, Bangalore, India."
            )
        ]

        # Wrap each text chunk into a LangChain Document object
        documents: List[Document] = [Document(page_content=text) for text in sample_texts]

        # Initialize RAG pipeline: stores QA chain and FAISS vectorstore in session state
        st.session_state.qa, st.session_state.vs = initialize_rag_pipeline(documents)
        st.success(f"Processed {len(documents)} sample chunks.")



    # If the pipeline is initialized, display the question UI and retrieval/response
    if "qa" in st.session_state and "vs" in st.session_state:
        st.subheader("Ask a question")
        user_query: str = st.text_input("Your question:")

        if st.button("Ask") and user_query:
            # Perform similarity search to show top-3 relevant chunks
            top_results: List[Tuple[Document, float]] = \
                st.session_state.vs.similarity_search_with_score(user_query, k=3)

            with st.expander("Relevant chunks"):
                for doc, score in top_results:
                    st.markdown(f"**Score:** {score:.4f}\n\n{doc.page_content}\n---")

            # Invoke the QA chain to get an answer based on retrieved context
            response = st.session_state.qa.invoke({"query": user_query})
            st.markdown(f"**Answer:** {response['result']}")

    # Sidebar section showing current configuration parameters
    with st.sidebar:
        st.write("**Configuration**")
        st.write(f"Model: {MODEL_NAME}")
        st.write(f"Temperature: {TEMPERATURE}")
        st.write(f"Chunk size: {CHUNK_SIZE}")
        st.write(f"Chunk overlap: {CHUNK_OVERLAP}")


if __name__ == "__main__":
    main()
