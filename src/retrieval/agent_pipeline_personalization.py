# retrieval/agent_pipeline.py

"""
retrieval/agent_pipeline.py

Defines a LangChain conversational Agent that uses FAISS-based document retrieval as a tool
and ChatOllama/ChatOpenAI as the underlying language model, with conversational memory support.

Functions:
- initialize_agent_pipeline: builds and returns the Agent and its vector store.
"""

from datetime import datetime
import logging
import os
from typing import List, Tuple

from langchain.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

from prompting.prompt_templates import SYSTEM_PROMPT, ACADEMIC_PROMPTS


from prompting.context_analyzer import ContextAnalyzer
from prompting.prompt_templates import get_template_key, format_prompt

# from langchain.chat_models import ChatOllama
# from langchain.embeddings import OllamaEmbeddings


from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings


# Import configuration constants
from config.settings import FAISS_INDEX_PATH, MODEL_NAME, TEMPERATURE 

# Initialize context analyzer globally
context_analyzer = ContextAnalyzer()


from utility.profile_manager import UserProfile

def personalize_prompt(base_prompt: str, user_profile: UserProfile) -> str:
    """Enhance prompt with user-specific context"""
    personalization = f"""
### PERSONALIZATION CONTEXT
User's Academic Interests: {', '.join(user_profile.data['academic_interests'])}
Preferred Style: {user_profile.data['preferred_style']}
Known Background: {user_profile.data['known_background']}
"""
    
    style_instructions = {
        "technical": "Include technical details and references",
        "simple": "Use layman's terms and simple analogies",
        "detailed": "Provide comprehensive explanations with examples"
    }.get(user_profile.data["preferred_style"], "")
    
    return f"{base_prompt}\n{personalization}\nStyle Instructions: {style_instructions}"

def process_query_with_personalization(agent, query: str, user_profile: UserProfile):
    user_profile.log_interaction(query, "")  # Log before response
    
    # Get base response
    response = agent.run(query)
    
    # Personalize post-processing
    if "computer science" in user_profile.data["academic_interests"]:
        response += "\n\n[Related CS Resources: ...]"
    
    user_profile.log_interaction(query, response)  # Log final response
    return response


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
 URGENT QUERY HANDLING:
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

def load_agent_pipeline() -> Tuple[object, FAISS]:
    """
    Load a persisted FAISS vector store and reconstruct the conversational Agent.

    Returns:
        agent: A LangChain Agent that uses the loaded FAISS index.
        vector_store: The FAISS store containing document embeddings.
    """
    # 1. Load embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # required due to pickle safety
    )

    # 2. Define retrieval tool
    def retrieve_docs(query: str) -> str:
        hits = vector_store.similarity_search_with_score(query, k=3)
        return "\n\n".join([doc.page_content for doc, _ in hits])

    retrieval_tool = Tool(
        name="faiss_retriever",
        func=retrieve_docs,
        description="Fetch top 3 relevant document chunks for a query."
    )

    # 3. Add a date tool (optional)
    def get_current_date(_query: str) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    date_tool = Tool(
        name="get_current_date",
        func=get_current_date,
        description="Returns the current date in YYYY-MM-DD format."
    )

    # 4. Set up conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 5. Create agent
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

    agent = initialize_agent(
        tools=[retrieval_tool, date_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent, vector_store

def initialize_agent_pipeline(
    documents: List[Document]
) -> Tuple[object, FAISS]:  # Agent is returned as a generic object
    """
    Instantiate a conversational Agent with FAISS retrieval and chat memory.

    Args:
        documents (List[Document]): List of text chunks to index in FAISS.

    Returns:
        agent: A LangChain Agent that will decide when to call retrieval.
        vector_store (FAISS): The FAISS store containing document embeddings.
    """
    # 1. Set up the language model and embeddings
    #llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)
    #embeddings = OllamaEmbeddings(model=MODEL_NAME)


    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    embeddings = OpenAIEmbeddings()


    # 2. Build FAISS index from supplied documents
    if os.path.exists(FAISS_INDEX_PATH):
        logging.info(" Loading existing FAISS index from disk.")
        vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
        )

        vector_store.add_documents(documents)
        logging.info(f"➕ Added {len(documents)} new documents to existing index.")
    else:
        logging.info("🆕 No FAISS index found. Creating new vector store.")
        vector_store = FAISS.from_documents(documents, embeddings)
        logging.info(f"✅ Created new index with {len(documents)} documents.")

    vector_store.save_local(FAISS_INDEX_PATH)
    logging.info(f"💾 Saved FAISS index to: {FAISS_INDEX_PATH}")


    # 3. Define a retrieval tool that returns concatenated top-3 documents
    def retrieve_docs(query: str) -> str:
        hits = vector_store.similarity_search_with_score(query, k=3)
        return "\n\n".join([doc.page_content for doc, _ in hits])

    retrieval_tool = Tool(
        name="faiss_retriever",
        func=retrieve_docs,
        description="Fetch the top 3 most relevant document chunks for a query."
    )

    #  Date tool 
    def get_current_date(_query: str) -> str:
        """Return today's date in YYYY-MM-DD format."""
        return datetime.now().strftime("%Y-%m-%d")

    date_tool = Tool(
        name="get_current_date",
        func=get_current_date,
        description="Returns the current date in YYYY-MM-DD format."
    )

    # 4. Configure conversational memory to keep chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 5. Initialize the agent with the retrieval tool and memory
    agent = initialize_agent(
        tools=[retrieval_tool, date_tool],
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True

    )
    

    return agent, vector_store

# Add to your existing agent_pipeline.py

def get_enhanced_retriever(vector_store):
    """Create enhanced retriever with better document formatting"""
    def retrieve_docs(query: str) -> str:
        docs = vector_store.similarity_search_with_score(query, k=3)
        return "\n\n---\n\n".join(
            f"DOCUMENT {i+1} (Relevance: {1-score:.2f}):\n{doc.page_content}"
            for i, (doc, score) in enumerate(docs)
        )
    return retrieve_docs

def create_agent_tools(vector_store):
    """Create all tools for the agent"""
    return [
        Tool(
            name="document_retriever",
            func=get_enhanced_retriever(vector_store),
            description="Searches academic documents for relevant information."
        ),
        Tool(
            name="current_date",
            func=lambda _: datetime.now().strftime("%Y-%m-%d"),
            description="Returns the current date when asked about dates or time."
        )
    ]

def create_agent(llm, tools, memory):
    """Initialize the conversational agent"""
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors="Check your output and try again.",
        max_iterations=5
    )
