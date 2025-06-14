from typing import List, Tuple, Dict, Any
import os

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph

from config.settings import FAISS_INDEX_PATH, MODEL_NAME, TEMPERATURE


def load_embeddings_and_store(documents: List[Document]) -> FAISS:
    """Load or create FAISS vector store with embeddings."""
    embeddings = OpenAIEmbeddings()

    if os.path.exists(FAISS_INDEX_PATH):
        vs = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        if documents:
            vs.add_documents(documents)
    else:
        vs = FAISS.from_documents(documents, embeddings)

    vs.save_local(FAISS_INDEX_PATH)
    return vs


def initialize_agent_pipeline(documents: List[Document]) -> Tuple[StateGraph, FAISS]:
    """Initialize the LangGraph agent pipeline with document retrieval capabilities."""
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    vs = load_embeddings_and_store(documents)

    # --- Node: Document Retriever ---
    def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant documents and generate answer."""
        query = state.get("query", "")
        docs = vs.similarity_search(query, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""You are a helpful academic assistant.

Context:
{context}

Question:
{query}
"""
        result = llm.invoke(prompt).content
        return {**state, "answer": result}

    # --- Node: Weekly Digest ---
    def weekly_digest_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate weekly digest of announcements and deadlines."""
        docs = vs.similarity_search("announcement OR deadline OR weekly", k=10)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""Summarize the following academic updates into:

📅 Deadlines  
📣 Announcements  
📝 Course Updates

Only include updates from the last 7 days.

TEXT:
{context}
"""
        result = llm.invoke(prompt).content
        return {**state, "answer": "📬 Weekly Digest:\n\n" + result}

    # --- Router Function ---
    def router(state: Dict[str, Any]) -> str:
        """Route to appropriate node based on query."""
        query = state.get("query", "").lower()
        if "weekly" in query or "digest" in query:
            return "weekly_digest"
        return "retriever"

    # --- Build LangGraph ---
    workflow = StateGraph(Dict[str, Any])
    
    # Add nodes
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("weekly_digest", weekly_digest_node)
    
    # Set entry point
    workflow.set_entry_point("retriever")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "retriever",
        router,
        {
            "retriever": END,  # Go to END after retrieval
            "weekly_digest": "weekly_digest"  # Only go to digest if needed
        }
    )
    
    # Weekly digest always ends after completion
    workflow.add_edge("weekly_digest", END)

    # Compile the graph
    graph = workflow.compile()
    return graph, vs


def load_agent_pipeline() -> Tuple[StateGraph, FAISS]:
    """Load the agent pipeline without adding new documents."""
    return initialize_agent_pipeline([])


# Example usage:
if __name__ == "__main__":
    # Initialize with sample documents if needed
    sample_docs = [
        Document(page_content="Assignment due Friday", metadata={"date": "2023-11-10"}),
        Document(page_content="Midterm exam next week", metadata={"date": "2023-11-15"})
    ]
    
    # Initialize or load pipeline
    graph, vectorstore = initialize_agent_pipeline(sample_docs)
    
    # Example queries
    result1 = graph.invoke({"query": "What are my upcoming deadlines?"})
    print("Regular query result:")
    print(result1["answer"])
    
    result2 = graph.invoke({"query": "Show me the weekly digest"})
    print("\nWeekly digest result:")
    print(result2["answer"])