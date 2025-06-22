from datetime import datetime
import logging
import os
from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from tools.gmail_tool import send_email
from tools.calender_invite import schedule_event

from config.settings import FAISS_INDEX_PATH, MODEL_NAME, TEMPERATURE
from prompting.prompt_templates import SYSTEM_PROMPT, ACADEMIC_PROMPTS


from prompting.context_analyzer import ContextAnalyzer
from prompting.prompt_templates import get_template_key, format_prompt

# from langchain.chat_models import ChatOllama
# from langchain.embeddings import OllamaEmbeddings


import requests
import json

def get_course_info_tool(course_code: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "method": "get_course_info",
        "params": {"course_code": course_code.strip()},
        "id": 1
    }
    try:
        response = requests.post(
            "http://localhost:5001",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        
        data = response.json()
        
        if "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
            error_detail = data["error"].get("data", "")
            return f" {error_msg}" + (f" ({error_detail})" if error_detail else "")
            
        result = data.get("result", {})
        if not result or "title" not in result:
            return " Course information not found or incomplete"
            
        return (
            f" {result['title']} (Credits: {result['credits']})\n"
            f" Instructor: {result['instructor']}\n"
            f" Semester: {result['semester']}"
        )
    except requests.exceptions.RequestException as e:
        return f" Connection error: {str(e)}"
    except json.JSONDecodeError:
        return " Invalid response from server"
    except Exception as e:
        return f" Unexpected error: {str(e)}"

# --- Weekly Digest Tool ---
def create_weekly_digest_tool(vector_store) -> Tool:
    def generate_digest(_query: str) -> str:
        logging.info("🧠 [Digest Tool] Generating weekly digest.")
        hits = vector_store.similarity_search("weekly update OR announcement OR deadline", k=10)
        combined = "\n\n".join([doc.page_content for doc in hits])
        prompt = f"""
Summarize the following recent academic updates into weekly digest format.
Group into: 📅 Deadlines, 📣 Announcements, 📝 Course-specific Updates.
Only include updates from the past 7 days.

TEXT:
{combined}
"""
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
        return "📬 (via `weekly_digest_agent`) Weekly Digest:\n\n" + llm.predict(prompt)

    return Tool(
        name="weekly_digest_agent",
        func=generate_digest,
        description="Summarize weekly academic updates like deadlines and announcements."
    )


# --- Personal Planner Tool ---
def create_personal_planner_tool(vector_store) -> Tool:
    def generate_plan(query: str) -> str:
        logging.info("🧠 [Planner Tool] Creating personalized academic plan.")
        hits = vector_store.similarity_search(query, k=6)
        content = "\n\n".join([doc.page_content for doc in hits])
        prompt = f"""
You are an academic planner. Based on the user's query and the provided text, identify any upcoming deadlines, especially related to specific courses like Machine Learning.

Only include deadlines from the current or next week (today: {datetime.now().strftime('%Y-%m-%d')}).

Respond clearly and only if a deadline is found.

---

User Query:
{query}

Retrieved Text:
{content}
"""

        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.4)
        return "📘 (via `personal_planner_agent`)\n\n" + llm.predict(prompt)

    return Tool(
        name="personal_planner_agent",
        func=generate_plan,
        description="Help students plan based on their course-specific deadlines, announcements, or progress queries."
    )


# --- Gmail Agent ---
def create_gmail_tool() -> Tool:
    def gmail_func(input: str) -> str:
        # Simple format: "to::subject::body"
        try:
            to, subject, body = input.split("::", 2)
            return send_email(to.strip(), subject.strip(), body.strip())
        except Exception as e:
            return f"Format: to::subject::body | Error: {e}"
    return Tool(
        name="send_gmail_agent",
        func=gmail_func,
        description="Send an email via Gmail. Format: to::subject::body"
    )
# --- Calendar Agent ---

def create_calendar_tool() -> Tool:
    def calendar_func(input: str) -> str:
        # Simple format: "summary::start_time::end_time::description"
        try:
            summary, start_time, end_time, description = input.split("::", 3)
            return schedule_event(summary.strip(), start_time.strip(), end_time.strip(), description.strip())
        except Exception as e:
            return f"Format: summary::start::end::description | Error: {e}"
    return Tool(
        name="schedule_calendar_event",
        func=calendar_func,
        description="Schedule a Google Calendar event. Format: summary::start_time::end_time::description"
    )
# Initialize context analyzer globally
context_analyzer = ContextAnalyzer()

def create_enhanced_retrieval_tool(vector_store: FAISS, context_analyzer: ContextAnalyzer):
    """
    Create an enhanced retrieval tool that uses context analysis for better retrieval
    """
    def enhanced_retrieve_docs(query: str) -> str:
        # Analyze query context
        context_analysis = context_analyzer.analyze_complete_context(query)
        
        # Enhanced retrieval based on context
        k = 5 if context_analysis.get("urgency") == "high" else 3
        
        # Get documents with scores
        hits = vector_store.similarity_search_with_score(query, k=k)
        
        # Format results with context awareness
        results = []
        for doc, score in hits:
            # Add relevance score for urgent queries
            if context_analysis.get("urgency") == "high":
                results.append(f"[Relevance: {1-score:.2f}] {doc.page_content}")
            else:
                results.append(doc.page_content)
        
        retrieved_content = "\n\n".join(results)
        
        # Log context for debugging
        logging.info(f"Query context: {context_analysis.get('intents', [])} | Urgency: {context_analysis.get('urgency')}")
        
        return retrieved_content

    return Tool(
        name="enhanced_faiss_retriever",
        func=enhanced_retrieve_docs,
        description="Fetch relevant document chunks with context-aware retrieval for academic queries about IISc Masters programs."
    )

def create_context_aware_agent_prompt(context_analysis: dict, original_query: str) -> str:
    """
    Create context-aware system prompt for the agent based on query analysis
    """
    base_prompt = SYSTEM_PROMPT
    
    # Add context-specific instructions
    context_additions = []
    
    if context_analysis.get("urgency") == "high":
        context_additions.append("""
🚨 URGENT QUERY HANDLING:
- This query requires immediate attention
- Prioritize time-sensitive information
- Include specific deadlines and contact information
- Use urgent language and clear action items
""")
    
    if context_analysis.get("clarification", {}).get("needs_clarification"):
        context_additions.append("""
CLARIFICATION NEEDED:
- The user's query is unclear or too vague
- Ask specific clarifying questions before retrieving information
- Provide options based on common student scenarios
- Be patient and helpful in guiding them to clarity
""")
    
    intents = context_analysis.get("intents", [])
    if "admission" in intents:
        context_additions.append("""
ADMISSION FOCUS:
- Prioritize admission deadlines and requirements
- Include eligibility criteria and application processes
- Mention document requirements and submission procedures
""")
    
    if "fees" in intents:
        context_additions.append("""
FINANCIAL INFORMATION FOCUS:
- Provide specific fee amounts and payment schedules
- Include scholarship and financial aid information
- Be sensitive to financial concerns
""")
    
    # Combine base prompt with context additions
    if context_additions:
        enhanced_prompt = base_prompt + "\n\nCONTEXT-SPECIFIC INSTRUCTIONS:\n" + "\n".join(context_additions)
    else:
        enhanced_prompt = base_prompt
    
    return enhanced_prompt

def process_query_with_context(agent, query: str, context_analyzer: ContextAnalyzer) -> str:
    """
    Process user query with context analysis and enhanced prompting
    """
    # Analyze query context
    context_analysis = context_analyzer.analyze_complete_context(query)
    
    # Handle clarification needed cases
    if context_analysis.get("clarification", {}).get("needs_clarification"):
        template_key = get_template_key(context_analysis)
        clarification_prompt = format_prompt(
            template_key=template_key,
            context_analysis=context_analysis,
            retrieved_context="",
            query=query
        )
        return clarification_prompt
    
    # For regular queries, let the agent handle with enhanced context
    try:
        # Create context-aware system message
        enhanced_system_prompt = create_context_aware_agent_prompt(context_analysis, query)
        
        # Modify the agent's system message temporarily
        original_system_message = None
        if hasattr(agent, 'agent') and hasattr(agent.agent, 'llm_chain'):
            # Store original system message if exists
            prompt_template = agent.agent.llm_chain.prompt
            if hasattr(prompt_template, 'messages') and prompt_template.messages:
                for i, msg in enumerate(prompt_template.messages):
                    if isinstance(msg, SystemMessage):
                        original_system_message = msg
                        # Update with enhanced prompt
                        prompt_template.messages[i] = SystemMessage(content=enhanced_system_prompt)
                        break
        
        # Run the agent with enhanced context
        response = agent.run(query)
        
        # Restore original system message
        if original_system_message and hasattr(agent, 'agent') and hasattr(agent.agent, 'llm_chain'):
            prompt_template = agent.agent.llm_chain.prompt
            if hasattr(prompt_template, 'messages'):
                for i, msg in enumerate(prompt_template.messages):
                    if isinstance(msg, SystemMessage):
                        prompt_template.messages[i] = original_system_message
                        break
        
        return response
        
    except Exception as e:
        logging.error(f"Error in context-aware query processing: {e}")
        # Fallback to regular agent processing
        return agent.run(query)

# --- Load Persistent Agent ---
def load_agent_pipeline() -> Tuple[object, FAISS]:
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Define tools
    def retrieve_docs(query: str) -> str:
        hits = vector_store.similarity_search_with_score(query, k=3)
        return "\n\n".join([doc.page_content for doc, _ in hits])

    retrieval_tool = Tool(
        name="faiss_retriever",
        func=retrieve_docs,
        description="Fetch top 3 relevant document chunks for a query."
    )

    mcp_course_tool = Tool(
    name="get_course_info",
    func=get_course_info_tool,
    description="""**EXCLUSIVELY** use this for ANY query containing:
    - Course codes (e.g., 'CS101', 'EE202')
    - Requests for course details (title, instructor, credits, schedule)
    - Questions like 'Who teaches X?', 'What is X about?', 'When is X offered?'
    
    **Input MUST be:**
    - An **exact** capitalized course code (e.g., 'CS101', not 'cs101')
    - Extracted **verbatim** from the user's query
    
    **Never use this for:**
    - General questions (use `faiss_retriever`)
    - Non-course-related queries"""
    )
    date_tool = Tool(
        name="get_current_date",
        func=lambda _: datetime.now().strftime("%Y-%m-%d"),
        description="Returns the current date in YYYY-MM-DD format."
    )

    weekly_digest_tool = create_weekly_digest_tool(vector_store)
    personal_planner_tool = create_personal_planner_tool(vector_store)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

    agent = initialize_agent(
        tools=[mcp_course_tool,retrieval_tool, date_tool, weekly_digest_tool, personal_planner_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent, vector_store


# --- Initialize Agent with New Documents ---
def initialize_agent_pipeline(documents: List[Document]) -> Tuple[object, FAISS]:
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    embeddings = OpenAIEmbeddings()

    if os.path.exists(FAISS_INDEX_PATH):
        logging.info("📁 Loading existing FAISS index from disk.")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(documents)
        logging.info(f"➕ Added {len(documents)} new documents to existing index.")
    else:
        logging.info("🆕 No FAISS index found. Creating new vector store.")
        vector_store = FAISS.from_documents(documents, embeddings)
        logging.info(f"✅ Created new index with {len(documents)} documents.")

    vector_store.save_local(FAISS_INDEX_PATH)
    logging.info(f"💾 Saved FAISS index to: {FAISS_INDEX_PATH}")

    # Tools
    def retrieve_docs(query: str) -> str:
        hits = vector_store.similarity_search_with_score(query, k=3)
        return "\n\n".join([doc.page_content for doc, _ in hits])

    retrieval_tool = Tool(
        name="faiss_retriever",
        func=retrieve_docs,
        description="Fetch top 3 relevant document chunks for a query."
    )

    date_tool = Tool(
        name="get_current_date",
        func=lambda _: datetime.now().strftime("%Y-%m-%d"),
        description="Returns the current date."
    )

    mcp_course_tool = Tool(
    name="get_course_info",
    func=get_course_info_tool,
    description="""**EXCLUSIVELY** use this for ANY query containing:
    - Course codes (e.g., 'CS101', 'EE202')
    - Requests for course details (title, instructor, credits, schedule)
    - Questions like 'Who teaches X?', 'What is X about?', 'When is X offered?'
    
    **Input MUST be:**
    - An **exact** capitalized course code (e.g., 'CS101', not 'cs101')
    - Extracted **verbatim** from the user's query
    
    **Never use this for:**
    - General questions (use `faiss_retriever`)
    - Non-course-related queries"""
    )

    gmail_tool = create_gmail_tool()
    calendar_tool = create_calendar_tool()

    weekly_digest_tool = create_weekly_digest_tool(vector_store)
    personal_planner_tool = create_personal_planner_tool(vector_store)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=[mcp_course_tool,retrieval_tool, date_tool, weekly_digest_tool, personal_planner_tool, gmail_tool, calendar_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent, vector_store
