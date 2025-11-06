import os
import streamlit as st
from langchain_groq import ChatGroq
import uuid # CRITICAL: For generating a unique thread ID
from langchain_core.tools import Tool #  Corrected import location for Tool
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import create_agent 
from langchain_core.messages import HumanMessage, AIMessage 
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()

# --- 1. Tool Setup ---
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Create the DuckDuckGo tool correctly to bypass import/validation issues.
# 1. Instantiate the API Wrapper (NO 'name' ARGUMENT HERE)
search_wrapper = DuckDuckGoSearchAPIWrapper() 

# 2. Wrap the API Wrapper's run function into a Tool (SET 'name' HERE)
search = Tool.from_function(
    func=search_wrapper.run,
    name="Search",
    description="A tool for performing general web searches using DuckDuckGo."
)

ALL_TOOLS = [search, arxiv, wiki]
SYSTEM_PROMPT = "You are a helpful, expert AI assistant with access to various external tools like web search, Arxiv, and Wikipedia. Use these tools when necessary to answer user questions, especially for real-time information or specific academic queries."

# --- 2. Streamlit UI Setup ---
st.title("🗨️Chat with 🔎 Search (Groq Agent)")
st.markdown("""
This application uses a LangChain agent running on Groq, leveraging `StreamlitCallbackHandler` to display the agent's thought process and tool use.
""")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# --- 3. Session State Initialization (CRITICAL FOR STATE AND HISTORY) ---

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you? "}
    ]

# Initialize thread ID ONLY ONCE for consistent history
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

# Initialize agent_executor ONLY ONCE when API key is provided
if "agent_executor" not in st.session_state and api_key:
    try:
        model = ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant", streaming=True)
        memory = MemorySaver()
        st.session_state["agent_executor"] = create_agent(
            model=model, 
            tools=ALL_TOOLS,
            system_prompt=SYSTEM_PROMPT, 
            checkpointer=memory,
        )
    except Exception as e:
        st.error(f"Failed to initialize ChatGroq or Agent: {e}. Please check your Groq API key.")
        st.session_state["agent_executor"] = None

# Display existing messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 4. Handle User Input and Agent Invocation ---

if prompt := st.chat_input(placeholder="what is machine learning?"):
    if not api_key or st.session_state.get("agent_executor") is None:
        st.warning("Please enter your Groq API Key and ensure the agent initialized.")
        st.stop()
        
    # 1. Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    agent_executor = st.session_state["agent_executor"]

    with st.chat_message("assistant"):
        st.subheader("Agent's Thought Process and Tool Use:")
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        
        try:
            # Convert the list of Streamlit dictionaries back into LangChain Message objects
            langchain_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            # Correct Agent Invocation
            response = agent_executor.invoke(
                {"messages": langchain_messages},
                config={
                    "callbacks": [st_cb],
                    "configurable": {"thread_id": st.session_state["thread_id"]} 
                }      
            )

            # Extract Final Content
            final_response_content = response["messages"][-1].content

            # 2. Display final response and update session state
            st.write("### Final Response:")
            st.write(final_response_content)
            st.session_state.messages.append({'role':"assistant", "content": final_response_content})

        except Exception as e:
            error_message = f"An error occurred during agent execution: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.write(error_message)

