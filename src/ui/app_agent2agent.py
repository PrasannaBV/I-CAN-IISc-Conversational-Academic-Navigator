import streamlit as st
import os
import csv
import io
import pandas as pd
import fitz

from ingestion.website_processor import fetch_and_process_website
from config.settings import MODEL_NAME, TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH
from retrieval.agent2agent import initialize_agent_pipeline, load_agent_pipeline
from langchain.docstore.document import Document


def extract_text_from_pdf(file) -> list[Document]:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return [Document(page_content=page.get_text().strip()) for page in doc if page.get_text().strip()]


def main():
    st.set_page_config(page_title="I-CAN Agent", layout="wide")
    st.title("🤖 I-CAN Agent: IISc Conversational Academic Navigator")

    if "agent" not in st.session_state or "vs" not in st.session_state:
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                agent, vs = load_agent_pipeline()
                st.session_state.agent = agent
                st.session_state.vs = vs
                st.info("✅ Multi-agent system loaded from vector store.")
            except Exception as e:
                st.error(f"❌ Failed to load agents: {e}")
        else:
            st.warning("⚠️ No vector store found. Please ingest some content using the sidebar.")

    col1, col2 = st.columns([1, 3])

    with col2:
        st.subheader("💬 Chat with the I-CAN Agent")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "feedback_history" not in st.session_state:
            st.session_state.feedback_history = []

        with st.form(key="chat_form", clear_on_submit=True):
            user_query = st.text_input("Type your question and press Enter:")
            submitted = st.form_submit_button("Send")

        if submitted and user_query:
            with st.spinner("Thinking..."):
                try:
                    current_answer = st.session_state.agent.run({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history
                    })

                    st.session_state.chat_history.append((user_query, current_answer))
                    st.session_state.feedback_history.append("🤷 Not sure")
                except Exception as e:
                    st.session_state.chat_history.append((user_query, f"❌ Error: {e}"))
                    st.session_state.feedback_history.append("🤷 Not sure")

        for idx, (user_msg, agent_msg) in enumerate(st.session_state.chat_history):
            st.markdown(f"**👤 You:** {user_msg}")
            st.markdown(f"**🤖 Agent:** {agent_msg}")

            feedback_key = f"feedback_{idx}"
            st.session_state.feedback_history[idx] = st.radio(
                "Was this response helpful?",
                ("👍 Yes", "👎 No", "🤷 Not sure"),
                index=("👍 Yes", "👎 No", "🤷 Not sure").index(st.session_state.feedback_history[idx]),
                key=feedback_key,
                horizontal=True
            )

        if st.session_state.feedback_history:
            st.markdown("### 📊 Feedback Summary")
            df = pd.DataFrame(st.session_state.feedback_history, columns=["Feedback"])
            counts = df["Feedback"].value_counts()
            st.write(counts.to_frame("Count"))

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

    with col1:
        with st.expander("📥 Tools", expanded=False):
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
                            else:
                                st.session_state.agent, st.session_state.vs = initialize_agent_pipeline(docs)
                            st.success(f"✅ Ingested {len(docs)} chunks.")
                        else:
                            st.error("❌ No content extracted.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.warning("Please enter a valid URL.")

            st.write("### 📝 Paste Text Ingestion")
            pasted_text = st.text_area("Enter text to ingest:")
            if st.button("Ingest Text"):
                if pasted_text.strip():
                    doc = Document(page_content=pasted_text)
                    try:
                        if "vs" in st.session_state:
                            st.session_state.vs.add_documents([doc])
                            st.session_state.vs.save_local(FAISS_INDEX_PATH)
                        else:
                            st.session_state.agent, st.session_state.vs = initialize_agent_pipeline([doc])
                        st.success("✅ Text added to vector store.")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.warning("Please enter non-empty text.")

            st.write("### 📄 PDF Ingestion")
            uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
            if st.button("Ingest PDF"):
                if uploaded_pdf:
                    try:
                        pdf_docs = extract_text_from_pdf(uploaded_pdf)
                        if "vs" in st.session_state:
                            st.session_state.vs.add_documents(pdf_docs)
                            st.session_state.vs.save_local(FAISS_INDEX_PATH)
                        else:
                            st.session_state.agent, st.session_state.vs = initialize_agent_pipeline(pdf_docs)
                        st.success(f"✅ Ingested {len(pdf_docs)} pages from PDF.")
                    except Exception as e:
                        st.error(f"❌ PDF processing failed: {e}")
                else:
                    st.warning("Please upload a PDF.")

            st.write("### ⚙️ Configuration")
            st.markdown(f"- **Model:** `{MODEL_NAME}`")
            st.markdown(f"- **Temperature:** `{TEMPERATURE}`")
            st.markdown(f"- **Chunk Size:** `{CHUNK_SIZE}`")
            st.markdown(f"- **Chunk Overlap:** `{CHUNK_OVERLAP}`")

            st.write("### 🔍 Last Retrieved Chunks")
            if "vs" in st.session_state and st.session_state.chat_history:
                last_query = st.session_state.chat_history[-1][0]
                docs = st.session_state.vs.similarity_search_with_score(last_query, k=5)
                for i, (doc, score) in enumerate(docs):
                    st.markdown(f"**[{i+1}] Score:** {score:.4f}")
                    st.markdown(doc.page_content)

            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.feedback_history = []
                st.success("✅ Chat cleared.")


if __name__ == "__main__":
    main()
