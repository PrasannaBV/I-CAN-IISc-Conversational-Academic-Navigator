
import streamlit as st
import os
import json
from pathlib import Path
import pandas as pd
import fitz
from datetime import datetime
from langchain.docstore.document import Document
from config.settings import MODEL_NAME, TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH
from retrieval.agent_pipeline import initialize_agent_pipeline, load_agent_pipeline
from prompting.prompt_manager import get_enhanced_query, get_followup_suggestions
from ingestion.website_processor import fetch_and_process_website
from utility.profile_manager import UserProfile

# Constants
MEMORY_PATH = Path("data/conversational_memory.json")
MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)


def initialize_session():
    """Initialize or restore session state"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.chat_history = load_chat_history()
        st.session_state.feedback_history = ["🤷 Not sure"] * len(st.session_state.chat_history)
        
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                agent, vs = load_agent_pipeline()
                st.session_state.agent = agent
                st.session_state.vs = vs
            except Exception as e:
                st.error(f"Failed to load agent: {e}")
                
# --- Core Personalization Functions ---
def personalize_response(response: str, profile: UserProfile) -> str:
    """Adapt responses based on user profile"""
    # Style adaptation
    style_map = {
        "technical": lambda x: f" Technical Perspective:\n{x}\n\n[Relevant Papers]",
        "simple": lambda x: f"Simplified Explanation:\n{simplify_text(x)}",
        "detailed": lambda x: f" Detailed Analysis:\n{x}\n\n[Examples] [Case Studies]"
    }
    
    # Apply style transformation
    response = style_map.get(profile.data["preferred_style"], lambda x: x)(response)
    
    # Add interest-based resources
    if profile.data["academic_interests"]:
        response += f"\n\nRelated to your interests in: {', '.join(profile.data['academic_interests'])}"
    
    # Department-specific additions
    if "department" in profile.data:
        dept_resources = {
            "Computer Science": "\n\nCS Resources: [Course Catalog] [Lab Contacts]",
            "Engineering": "\n\nEngineering Links: [Faculty Directory]"
        }
        response += dept_resources.get(profile.data["department"], "")
    
    return response

def simplify_text(text: str) -> str:
    """Convert complex terms to simpler language"""
    replacements = {
        "pedagogical": "teaching",
        "curriculum": "course plan",
        "prerequisites": "requirements needed before"
    }
    for term, replacement in replacements.items():
        text = text.replace(term, replacement)
    return text

# --- Modified Core Functions ---
import streamlit as st
from utility.profile_manager import UserProfile

# --- 1. Modified Core Functions ---
def handle_user_query(user_query: str):
    """Process query with personalization and ensure UI updates"""
    if 'user_profile' not in st.session_state:
        st.error("Please complete your profile first")
        return None

    # Clear any previous error messages
    st.session_state.error_message = None
    
    try:
        with st.spinner("Crafting your personalized answer..."):
            # 1. Get context documents
            docs = st.session_state.vs.similarity_search_with_score(user_query, k=5)
            
            # 2. Enhance query
            enhanced_query = get_enhanced_query(user_query, docs)
            
            # 3. Get base response
            base_response = st.session_state.agent.run(enhanced_query)
            
            # 4. Personalize 
            response = personalize_response(base_response, st.session_state.user_profile)
            
            # 5. Update session state 
            st.session_state.chat_history.append((user_query, response))
            st.session_state.feedback_history.append("🤷 Not sure")
            
            # 6. Generate follow-ups
            followups = get_followup_suggestions(user_query, response)
            st.session_state.followup_suggestions = followups 
            
            # 7. Persist data
            save_chat_history(st.session_state.chat_history)
            st.session_state.user_profile.save()
            
            # Force UI update
            st.rerun()
            
    except Exception as e:
        st.session_state.error_message = f"Error generating response: {str(e)}"
        st.rerun()
# --- Profile Management UI ---
def show_profile_editor():
    """UI for profile customization"""
    with st.sidebar.expander("👤 Your Profile"):
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = UserProfile("default")
            
        profile = st.session_state.user_profile.data
        
        # 1. User Type Selection 
        user_types = ["Prospective Student", "Current Student", "Researcher", "Faculty", "Alumni"]
        current_user_type = profile.get("user_type", "Prospective Student")
        try:
            user_type_index = user_types.index(current_user_type)
        except ValueError:
            user_type_index = 0
            
        profile["user_type"] = st.selectbox(
            "You are a:",
            user_types,
            index=user_type_index
        )
        
        # 2. Department Selection
        departments = ["Computer Science", "Engineering", "Mathematics", "Physics", "Other"]
        current_dept = profile.get("department", "Computer Science")
        try:
            dept_index = departments.index(current_dept)
        except ValueError:
            dept_index = 0
            
        profile["department"] = st.selectbox(
            "Department Interest:",
            departments,
            index=dept_index
        )
        
        # 3. Style Preference
        styles = ["technical", "simple", "detailed"]
        current_style = profile.get("preferred_style", "technical")
        try:
            style_index = styles.index(current_style)
        except ValueError:
            style_index = 0
            
        profile["preferred_style"] = st.radio(
            "Preferred response style:",
            styles,
            index=style_index
        )
        
        # 4. Academic Interests
        new_interest = st.text_input("Add Academic Interest")
        if st.button("Add") and new_interest:
            st.session_state.user_profile.update_interest(new_interest)
            st.rerun()
        
        # Display current interests with removal option
        if profile.get("academic_interests"):
            st.write("Current Interests:")
            cols = st.columns(3)
            for i, interest in enumerate(profile["academic_interests"]):
                with cols[i % 3]:
                    if st.button(f"❌ {interest}", key=f"del_interest_{i}"):
                        profile["academic_interests"].remove(interest)
                        st.session_state.user_profile.save()
                        st.rerun()
        
        # 5. Background Info
        profile["known_background"] = st.text_area(
            "Your academic/professional background",
            profile.get("known_background", "")
        )
        
        if st.button("💾 Save Profile"):
            st.session_state.user_profile.save()
            st.success("Profile updated!")




def save_chat_history(chat_history):
    """Save chat history to JSON file"""
    try:
        with open(MEMORY_PATH, 'w') as f:
            json.dump(chat_history, f)
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")

def load_chat_history():
    """Load chat history from JSON file"""
    try:
        if MEMORY_PATH.exists():
            with open(MEMORY_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Failed to load chat history: {e}")
    return []

def extract_text_from_pdf(file) -> list[Document]:
    """Extracts text from PDF file object and returns a list of Documents."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    documents = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            documents.append(Document(page_content=text))
    return documents

def initialize_session():
    """Initialize or restore session state"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.chat_history = load_chat_history()
        st.session_state.feedback_history = ["🤷 Not sure"] * len(st.session_state.chat_history)
        
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                agent, vs = load_agent_pipeline()
                st.session_state.agent = agent
                st.session_state.vs = vs
            except Exception as e:
                st.error(f"Failed to load agent: {e}")


def render_chat_interface():
    """Render the chat with guaranteed updates"""
    st.subheader("💬 Personalized Chat")
    
    # Display error if exists
    if hasattr(st.session_state, 'error_message') and st.session_state.error_message:
        st.error(st.session_state.error_message)
    
    # Display chat history
    for idx, (user_msg, agent_msg) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(agent_msg)
            
            # Show follow-ups only for last message
            if idx == len(st.session_state.chat_history) - 1:
                show_followup_suggestions(idx)
        
        render_feedback_section(idx)
        st.divider()

def show_followup_suggestions(idx):
    """Display follow-up suggestions for a message"""
    if ('followup_suggestions' in st.session_state and 
        idx < len(st.session_state.followup_suggestions)):
        followups = st.session_state.followup_suggestions[idx]
        if followups:
            st.markdown("**💡 Suggested follow-ups:**")
            cols = st.columns(min(2, len(followups)))
            for i, suggestion in enumerate(followups[:4]):
                with cols[i % 2]:
                    if st.button(suggestion, key=f"followup_{idx}_{i}"):
                        st.session_state.selected_followup = suggestion
                        st.rerun()

def render_feedback_section(idx):
    """Render feedback UI for a message"""
    feedback_key = f"feedback_{idx}"
    st.session_state.feedback_history[idx] = st.radio(
        "Was this helpful?",
        ("👍 Yes", "👎 No", "🤷 Not sure"),
        index=2,
        key=feedback_key,
        horizontal=True
    )

def render_sidebar():
    """Render the sidebar content"""
    with st.sidebar:
        st.title("⚙️ Tools & Settings")
        
        with st.expander("📥 Content Ingestion"):
            render_ingestion_controls()
            
        with st.expander("🔍 Retrieved Chunks"):
            render_retrieved_chunks()
            
        with st.expander("📊 Statistics"):
            render_statistics()
            
        if st.button("🧹 Clear Chat History"):
            clear_chat_history()

def render_ingestion_controls():
    """Render content ingestion controls"""
    # Website ingestion
    url = st.text_input("Website URL:")
    if st.button("Process Website"):
        process_website(url)
        
    # Text ingestion
    text = st.text_area("Paste Text:")
    if st.button("Ingest Text"):
        process_text(text)
        
    # PDF ingestion
    pdf = st.file_uploader("Upload PDF:", type=["pdf"])
    if st.button("Ingest PDF") and pdf:
        process_pdf(pdf)

def process_website(url):
    """Process website content"""
    if url:
        try:
            docs = fetch_and_process_website(url)
            update_vector_store(docs, f"Processed website: {url}")
        except Exception as e:
            st.error(f"Error processing website: {e}")

def process_text(text):
    """Process pasted text"""
    if text.strip():
        try:
            docs = [Document(page_content=text)]
            update_vector_store(docs, "Processed text")
        except Exception as e:
            st.error(f"Error processing text: {e}")

def process_pdf(pdf):
    """Process uploaded PDF"""
    try:
        docs = extract_text_from_pdf(pdf)
        update_vector_store(docs, f"Processed PDF: {pdf.name}")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")

def update_vector_store(docs, success_msg):
    """Update the vector store with new documents"""
    if docs:
        if "vs" in st.session_state:
            st.session_state.vs.add_documents(docs)
            st.session_state.vs.save_local(FAISS_INDEX_PATH)
        else:
            st.session_state.agent, st.session_state.vs = initialize_agent_pipeline(docs)
        st.success(f"✅ {success_msg}")

def render_retrieved_chunks():
    """Display retrieved document chunks"""
    if "vs" in st.session_state and st.session_state.chat_history:
        last_query = st.session_state.chat_history[-1][0]
        docs = st.session_state.vs.similarity_search_with_score(last_query, k=3)
        for doc, score in docs:
            st.markdown(f"**Relevance:** {1-score:.2f}")
            st.text(doc.page_content[:200] + "...")

def render_statistics():
    """Display chat statistics"""
    if st.session_state.chat_history:
        st.metric("Total Messages", len(st.session_state.chat_history))
        
        if st.session_state.feedback_history:
            feedback_counts = pd.Series(st.session_state.feedback_history).value_counts()
            st.dataframe(feedback_counts.rename("Count"))

def clear_chat_history():
    """Clear the chat history"""
    st.session_state.chat_history = []
    st.session_state.feedback_history = []
    if 'followup_suggestions' in st.session_state:
        del st.session_state.followup_suggestions
    save_chat_history([])
    st.success("Chat history cleared")


# --- 3.  Main Function ---
def main():
    st.set_page_config(page_title="I-CAN Agent", layout="wide")
    st.title("🤖 Personalized Academic Assistant")
    
    # Initialize critical components
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = UserProfile("default")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history() or []
    
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = ["🤷 Not sure"] * len(st.session_state.chat_history)
    
    initialize_session()
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Profile completion check
        if not st.session_state.user_profile.data.get("user_type"):
            st.warning("Please complete your profile to continue")
            show_profile_editor()
        else:
            render_chat_interface()
            
            # Chat input with guaranteed processing
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input("Your question:", key="user_input")
                submitted = st.form_submit_button("Get Personalized Answer")
                
                if submitted and user_input:
                    handle_user_query(user_input)
    
    with col2:
        show_profile_editor()
        render_sidebar()

if __name__ == "__main__":
    main()


