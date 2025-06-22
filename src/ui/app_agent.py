from ingestion.website_processor import fetch_and_process_website
import streamlit as st
import os
import csv
import io
import pandas as pd
import fitz  # to chunk pdf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import MODEL_NAME, TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH
from retrieval.agent_pipeline import initialize_agent_pipeline, load_agent_pipeline
from langchain.docstore.document import Document

from prompting.prompt_manager import get_enhanced_query, get_followup_suggestions


def extract_text_from_pdf(file) -> list[Document]:
    """Extracts text from PDF file object and returns a list of Documents."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    documents = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            documents.append(Document(page_content=text))
    return documents


def main() -> None:
    st.set_page_config(page_title="I-CAN Agent", layout="wide")
    st.title("🤖 I-CAN Agent: IISc Conversational Academic Navigator")

    if "agent" not in st.session_state or "vs" not in st.session_state:
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                agent, vs = load_agent_pipeline()
                st.session_state.agent = agent
                st.session_state.vs = vs
                st.info("✅ Agent loaded from persistent vector store.")
            except Exception as e:
                st.error(f"❌ Failed to load agent: {e}")
        else:
            st.warning("⚠️ No vector store found. Please ingest some content using the sidebar.")

    # Columns
    col1, col2 = st.columns([1, 3])

    # --- Chat Area ---
    with col2:
        st.subheader("💬 Chat with the I-CAN Agent")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "feedback_history" not in st.session_state:
            st.session_state.feedback_history = []

        current_answer = None

        with st.form(key="chat_form", clear_on_submit=True):
            user_query = st.text_input("Type your question and press Enter:")
            submitted = st.form_submit_button("Send")

        # Handle follow-up question selection
        if 'selected_followup' in st.session_state:
            user_query = st.session_state.selected_followup
            del st.session_state.selected_followup  # Clear the selection
            submitted = True  # Trigger processing

        if submitted and user_query and "agent" in st.session_state:
            with st.spinner("Thinking..."):
                # Get relevant documents
                docs = st.session_state.vs.similarity_search_with_score(user_query, k=5)
                
                # Use prompt manager to enhance the query with context and appropriate prompting
                enhanced_query = get_enhanced_query(user_query, docs)
                
                # Get response from agent using enhanced query
                current_answer = st.session_state.agent.run(enhanced_query)
                
                # Store the conversation
                st.session_state.chat_history.append((user_query, current_answer))
                st.session_state.feedback_history.append("🤷 Not sure")
                
                # Get follow-up suggestions and store them
                followups = get_followup_suggestions(user_query, current_answer)
                if 'followup_suggestions' not in st.session_state:
                    st.session_state.followup_suggestions = []
                st.session_state.followup_suggestions.append(followups)

        # Display chat + feedback with follow-up suggestions
        for idx, (user_msg, agent_msg) in enumerate(st.session_state.chat_history):
            st.markdown(f"**👤 You:** {user_msg}")
            st.markdown(f"**🤖 Agent:** {agent_msg}")
            
            # Show follow-up suggestions for the latest message
            if (idx == len(st.session_state.chat_history) - 1 and 
                'followup_suggestions' in st.session_state and 
                idx < len(st.session_state.followup_suggestions)):
                
                followups = st.session_state.followup_suggestions[idx]
                if followups:
                    st.markdown("**💡 Suggested follow-up questions:**")
                    cols = st.columns(min(len(followups), 2))  # Max 2 columns
                    for i, suggestion in enumerate(followups[:4]):  # Show max 4 suggestions
                        col_idx = i % 2
                        with cols[col_idx]:
                            if st.button(f"💬 {suggestion}", key=f"followup_{idx}_{i}", use_container_width=True):
                                # Trigger a rerun with the selected suggestion
                                st.session_state.selected_followup = suggestion
                                st.rerun()

            # Feedback section
            feedback_key = f"feedback_{idx}"
            st.session_state.feedback_history[idx] = st.radio(
                "Was this response helpful?",
                ("👍 Yes", "👎 No", "🤷 Not sure"),
                index=("👍 Yes", "👎 No", "🤷 Not sure").index(st.session_state.feedback_history[idx]),
                key=feedback_key,
                horizontal=True
            )
            
            st.markdown("---")  # Separator between conversations

        # 📊 Feedback Summary
        if st.session_state.feedback_history:
            st.markdown("### 📊 Feedback Summary")
            df = pd.DataFrame(st.session_state.feedback_history, columns=["Feedback"])
            counts = df["Feedback"].value_counts()
            st.write(counts.to_frame("Count"))

            # 📁 Export to CSV
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["Question", "Answer", "Feedback"])
            for (q, a), fb in zip(st.session_state.chat_history, st.session_state.feedback_history):
                writer.writerow([q, a, fb])
            st.download_button(
                label="📥 Download Feedback as CSV",
                data=csv_buffer.getvalue(),
                file_name="feedback_history.csv",
                mime="text/csv"
            )

    # --- Sidebar: Ingestion + Config + Retrieved Chunks ---
    with col1:
        with st.expander("📥 Tools", expanded=False):
            #st.header("📥 Ingestion Panel")

            # Website ingestion
            st.write("### 🌐 Website Ingestion")
            url = st.text_input("Enter website URL:")
            if st.button("Process Website"):
                if url:
                    try:
                        docs = fetch_and_process_website(url)
                        if docs:
                            if "vs" in st.session_state:
                                st.session_state.vs.add_documents(docs)
                                st.session_state.vs.save_local(FAISS_INDEX_PATH)
                                st.success(f"✅ Added {len(docs)} chunks to existing vector store.")
                            else:
                                st.session_state.agent, st.session_state.vs = initialize_agent_pipeline(docs)
                                st.success(f"✅ Initialized pipeline with {len(docs)} chunks from {url}")
                        else:
                            st.error("❌ No content extracted.")
                    except Exception as e:
                        st.error(f"❌ Error processing site: {e}")
                else:
                    st.warning("Please enter a valid URL.")

            # Text ingestion
            st.write("### 📝 Paste Text Ingestion")
            pasted_text = st.text_area("Enter text to ingest:")
            if st.button("Ingest Text"):
                if pasted_text.strip():
                    doc = Document(page_content=pasted_text)
                    try:
                        if "vs" in st.session_state:
                            st.session_state.vs.add_documents([doc])
                            st.session_state.vs.save_local(FAISS_INDEX_PATH)
                            st.success("✅ Text added to vector store.")
                        else:
                            st.session_state.agent, st.session_state.vs = initialize_agent_pipeline([doc])
                            st.success("✅ Initialized pipeline with pasted text.")
                    except Exception as e:
                        st.error(f"❌ Error ingesting text: {e}")
                else:
                    st.warning("Please enter non-empty text.")

            # PDF ingestion
            st.write("### 📄 PDF Ingestion")
            uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
            if st.button("Ingest PDF"):
                if uploaded_pdf:
                    try:
                        pdf_docs = extract_text_from_pdf(uploaded_pdf)
                        if "vs" in st.session_state:
                            st.session_state.vs.add_documents(pdf_docs)
                            st.session_state.vs.save_local(FAISS_INDEX_PATH)
                            st.success(f"✅ Ingested {len(pdf_docs)} pages from PDF.")
                        else:
                            st.session_state.agent, st.session_state.vs = initialize_agent_pipeline(pdf_docs)
                            st.success(f"✅ Pipeline initialized with {len(pdf_docs)} pages from uploaded PDF.")
                    except Exception as e:
                        st.error(f"❌ Failed to process PDF: {e}")
                else:
                    st.warning("Please upload a PDF file.")

            # Config info
            st.write("### ⚙️ Configuration")
            st.markdown(f"- **Model:** `{MODEL_NAME}`")
            st.markdown(f"- **Temperature:** `{TEMPERATURE}`")
            st.markdown(f"- **Chunk Size:** `{CHUNK_SIZE}`")
            st.markdown(f"- **Chunk Overlap:** `{CHUNK_OVERLAP}`")

            # 🔍 Retrieved Chunks
            st.write("### 🔍 Last Retrieved Chunks")
            if "vs" in st.session_state and "chat_history" in st.session_state and st.session_state.chat_history:
                last_query = st.session_state.chat_history[-1][0]
                docs = st.session_state.vs.similarity_search_with_score(last_query, k=5)
                for i, (doc, score) in enumerate(docs):
                    st.markdown(f"**[{i+1}] Score:** {score:.4f}")
                    st.markdown(doc.page_content)

            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.feedback_history = []
                if 'followup_suggestions' in st.session_state:
                    st.session_state.followup_suggestions = []
                st.success("✅ Chat history and feedback cleared.")
                #st.experimental_rerun()  # Force UI to refresh immediately


if __name__ == "__main__":
    main()
