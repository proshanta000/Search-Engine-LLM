import os
import uuid
import streamlit as st
from dotenv import load_dotenv

# LangChain/Groq Imports
from langchain_groq import ChatGroq
from langchain_community.tools import tool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.callbacks import StreamlitCallbackHandler
from langgraph.prebuilt import create_react_agent 
from langchain_core.messages import HumanMessage, AIMessage 
from langgraph.checkpoint.memory import MemorySaver 
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# --- CONFIGURATION CONSTANTS ---
MODEL_NAME = "llama-3.1-8b-instant"
TOOL_CONTENT_CHARS_MAX = 400 

# --- 1. TOOL INITIALIZATION (UNTOUCHED LOGIC) ---
def initialize_tools():
    class WikipediaInput(BaseModel):
        query: str = Field(description="Search Query for wikipedia")
    class ArxivInput(BaseModel):
        query: str = Field(description="Search Query for Arxiv")
    class DuckDuckGoSearchInput(BaseModel):
        query: str = Field(description="Search Query for DuckDuckGo")

    @tool("duckduckgo_search", args_schema=DuckDuckGoSearchInput)
    def duckduckgo_search(query: str):
        """Search DuckDuckGo for a given query and return the summary."""
        _duckduckgo_api_wrapper_instance = DuckDuckGoSearchAPIWrapper()
        return _duckduckgo_api_wrapper_instance.run(query)

    @tool("wikipedia_search", args_schema=WikipediaInput)
    def wikipedia_search(query: str):
        """Search wikipedia for a given query and return the summary."""
        _wikipedia_api_wrapper_instance = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
        return _wikipedia_api_wrapper_instance.run(query)

    @tool("arxiv_search", args_schema=ArxivInput)
    def arxiv_search(query: str):
        """Search Arxiv for a given query and return the summary."""
        _arxiv_api_wrapper_instance = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
        return _arxiv_api_wrapper_instance.run(query)

    return [arxiv_search, wikipedia_search, duckduckgo_search]

# --- 2. AGENT INITIALIZATION (UNTOUCHED LOGIC) ---
def initialize_agent_executor(api_key: str, tools: list):
    if not api_key:
        st.error("Groq API Key is missing.")
        return None
    try:
        model = ChatGroq(groq_api_key=api_key, model=MODEL_NAME, streaming=True)
        memory = MemorySaver()
        SYSTEM_PROMPT = (
            "You are a helpful, expert AI assistant with access to various external tools "
            "like web search, Arxiv, and Wikipedia. Use these tools when necessary."
        )
        agent_executor = create_react_agent(
            model=model, 
            tools=tools, 
            system_prompt=SYSTEM_PROMPT,
            checkpointer=memory,
        )
        st.session_state["agent_executor"] = agent_executor
        return agent_executor
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        return None

# --- 3. HELPER FUNCTIONS ---
def convert_messages_to_langchain(messages: list) -> list:
    langchain_messages = []
    for msg in messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
    return langchain_messages

# --- 4. STREAMLIT APP LOGIC (BLENDED WITH NEW UI) ---
def main():
    # --- PAGE CONFIG ---
    st.set_page_config(page_title="ScholarGroq AI", page_icon="🔎", layout="wide")

    # --- CUSTOM STYLING ---
    st.markdown("""
        <style>
            .stChatMessage { border-radius: 12px; margin-bottom: 15px; }
            .stStatusWidget { border: 1px solid #e0e0e0; border-radius: 10px; }
            .main-header { font-size: 2.5rem; font-weight: 700; color: #1E1E1E; margin-bottom: 0px; }
            .sub-header { font-size: 1rem; color: #666; margin-bottom: 20px; }
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR SETTINGS ---
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        default_key = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input("Groq API Key:", type="password", value=default_key)
        
        st.markdown("---")
        st.markdown("### 🛠️ Agent Capabilities")
        st.info("Searching: DuckDuckGo, Wikipedia, Arxiv Research")
        
        if st.button("🗑️ Clear Conversation"):
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with your research today?"}]
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.rerun()

    # --- HEADER ---
    st.markdown('<p class="main-header">🔎 ScholarGroq</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Autonomous Agent for Real-Time & Academic Discovery</p>', unsafe_allow_html=True)

    # --- STATE INITIALIZATION ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm an expert AI who can use web search, Arxiv, and Wikipedia. How can I help you?"}
        ]
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
    if "tools" not in st.session_state:
        st.session_state["tools"] = initialize_tools()

    if api_key and ("agent_executor" not in st.session_state or st.session_state.get("agent_executor") is None):
        st.session_state["agent_executor"] = initialize_agent_executor(api_key, st.session_state["tools"])

    # --- CHAT DISPLAY ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # --- USER INPUT & EXECUTION ---
    if prompt := st.chat_input(placeholder="e.g. Find recent breakthroughs in Quantum Computing on Arxiv."):
        
        agent_executor = st.session_state.get("agent_executor")
        if not api_key or agent_executor is None:
            st.warning("⚠️ Please provide a valid Groq API Key in the sidebar.")
            st.stop()
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            # UI Improvement: Wrap the thought process in an expandable status widget
            with st.status("🚀 Thinking and Searching...", expanded=True) as status:
                thought_container = st.container() 
                st_cb = StreamlitCallbackHandler(thought_container, expand_new_thoughts=True)
                
                langchain_messages = convert_messages_to_langchain(st.session_state.messages)
                
                try:
                    response = agent_executor.invoke(
                        {"messages": langchain_messages},
                        config={
                            "callbacks": [st_cb],
                            "configurable": {"thread_id": st.session_state["thread_id"]}
                        }
                    )
                    final_response_content = response["messages"][-1].content
                    status.update(label="✅ Research Complete", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    final_response_content = f"An error occurred: {e}"

            # Display final Markdown response
            st.markdown(final_response_content)
            st.session_state.messages.append({'role':"assistant", "content": final_response_content})

if __name__ == "__main__":
    main()