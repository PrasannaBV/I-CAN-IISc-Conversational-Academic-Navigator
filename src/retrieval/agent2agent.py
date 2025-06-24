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

from config.settings import FAISS_INDEX_PATH, MODEL_NAME, TEMPERATURE

# --- Digest Agent ---
def create_digest_agent(vector_store):
    def generate_digest(_: str) -> str:
        hits = vector_store.similarity_search("weekly update OR announcement OR deadline", k=10)
        combined = "\n\n".join([doc.page_content for doc in hits])
        prompt = f"""
Summarize the following recent academic updates into weekly digest format.
Group into: 🗕 Deadlines, 📣 Announcements, 🕘 Course-specific Updates.
Only include updates from the past 7 days.

TEXT:
{combined}
"""
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
        summary = llm.predict(prompt)
        return "📬 [Digest Agent]\n\n" + summary

    digest_tool = Tool(
        name="digest_tool",
        func=generate_digest,
        description="Summarize weekly academic updates."
    )

    return initialize_agent(
        tools=[digest_tool],
        llm=ChatOpenAI(model_name=MODEL_NAME),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

# --- Planner Agent ---
def create_planner_agent(vector_store):
    def generate_plan(query: str) -> str:
        hits = vector_store.similarity_search(query, k=6)
        content = "\n\n".join([doc.page_content for doc in hits])
        prompt = f"""
You are an academic planner. Based on the user's query and the provided text, identify any upcoming deadlines,
especially for specific courses. Only include deadlines for this week or next.
Today is {datetime.now().strftime('%Y-%m-%d')}.

Query:
{query}

TEXT:
{content}
"""
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.4)
        plan = llm.predict(prompt)
        return "📘 [Planner Agent]\n\n" + plan

    planner_tool = Tool(
        name="planner_tool",
        func=generate_plan,
        description="Help students plan based on course deadlines or queries."
    )

    return initialize_agent(
        tools=[planner_tool],
        llm=ChatOpenAI(model_name=MODEL_NAME),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

# --- Orchestrator Agent ---
def create_orchestrator_agent(digest_agent, planner_agent) -> object:
    def route_query(input_data) -> str:
        if isinstance(input_data, str):
            query = input_data
        elif isinstance(input_data, dict):
            query = input_data.get("input", "").strip()
        else:
            return "⚠️ Unexpected input format."

        print(f"🔁 Routing query: {query}")

        if "digest" in query.lower() or "summary" in query.lower():
            return digest_agent.run({"input": query})
        elif "deadline" in query.lower() or "plan" in query.lower():
            return planner_agent.run({"input": query})
        else:
            return "🤖 I'm not sure which agent should handle this. Try keywords like 'digest' or 'deadline'."

    delegator_tool = Tool(
        name="router",
        func=route_query,
        description="Routes user queries to digest or planner agent."
    )

    return initialize_agent(
        tools=[delegator_tool],
        llm=ChatOpenAI(model_name=MODEL_NAME),
        #agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=True,
        handle_parsing_errors=True
    )

# --- Load Existing Pipeline ---
def load_agent_pipeline() -> Tuple[object, FAISS]:
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    digest_agent = create_digest_agent(vector_store)
    planner_agent = create_planner_agent(vector_store)
    orchestrator = create_orchestrator_agent(digest_agent, planner_agent)

    return orchestrator, vector_store

# --- Initialize Pipeline with Documents ---
def initialize_agent_pipeline(documents: List[Document]) -> Tuple[object, FAISS]:
    embeddings = OpenAIEmbeddings()

    if os.path.exists(FAISS_INDEX_PATH):
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_documents(documents)
    else:
        vector_store = FAISS.from_documents(documents, embeddings)

    vector_store.save_local(FAISS_INDEX_PATH)

    digest_agent = create_digest_agent(vector_store)
    planner_agent = create_planner_agent(vector_store)
    orchestrator = create_orchestrator_agent(digest_agent, planner_agent)

    return orchestrator, vector_store
