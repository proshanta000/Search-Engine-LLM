import os
import streamlit as st
from langchain_groq import ChatGroq
import uuid

from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import create_agent 
from langchain_core.messages import HumanMessage, AIMessage 
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()

# --- 1. Tool Setup (UNCHANGED) ---
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")
ALL_TOOLS = [search, arxiv, wiki]
SYSTEM_PROMPT = "You are a helpful, expert AI assistant with access to various external tools like web search, Arxiv, and Wikipedia. Use these tools when necessary to answer user questions, especially for real-time information or specific academic queries."

# --- 2. Streamlit UI Setup ---
st.title("🗨️Chat with 🔎 Search")
st.markdown("""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of the agent in an interactive Streamlit app.
""")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# --- 3. Session State Initialization (CRITICAL FIXES) ---

# Initialize messages list
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you? "}
    ]

# Initialize thread ID ONLY ONCE for consistent history
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

# Initialize the agent executor outside the prompt loop (Fixes Issue 2)
if "agent_executor" not in st.session_state and api_key:
    # Initialize the model and agent only if API key is present
    try:
        model = ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant", streaming=True)
        memory = MemorySaver()
        st.session_state["agent_executor"] = create_agent(
            model=model, 
            tools=ALL_TOOLS,
            system_prompt=SYSTEM_PROMPT, # Use a static system prompt (Fixes Issue 4)
            checkpointer=memory,
        )
    except Exception as e:
        st.error(f"Failed to initialize ChatGroq or Agent: {e}")
        st.session_state["agent_executor"] = None

# Display existing messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# --- 4. Handle User Input and Agent Invocation ---
if prompt := st.chat_input(placeholder="What is machine learning?"):
    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar to begin chatting.")
        st.stop()
    
    if st.session_state.get("agent_executor") is None:
        st.error("Agent failed to initialize. Please check your API key and try refreshing.")
        st.stop()
        
    # 1. Store and display user message (Once)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt) # ⬅️ Only display here
    
    agent_executor = st.session_state["agent_executor"]

    with st.spinner("Generating response...."):
        
        # ADDITION: Explicitly label the area where the calculations/thoughts will appear.
        st.subheader("Agent's Thought Process and Tool Use:")
        # Setup callback handler. expand_new_thoughts=True ensures tool calls and outputs are visible.
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True) 
        
        try:
            # Convert the list of message dictionaries back into a list of 
            # LangChain Message objects (HumanMessage/AIMessage) for agent input.
            langchain_messages = []
            # Start from the *second* message to skip the initial "Hi, I'm a chatbot..." message 
            # if you only want the agent to use actual conversation history for context.
            # However, for simplicity and to include the welcome message in history, we'll convert all:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            response = agent_executor.invoke(
                {"messages": langchain_messages}, 
                config={
                    "callbacks": [st_cb],
                    # Pass the required thread_id to the checkpointer
                    "configurable": {"thread_id": st.session_state["thread_id"]} 
                }      
            )

            # Extract Final Content
            final_response_content = response["messages"][-1].content
            
            # 2. Update session state and display agent response
            with st.chat_message("assistant"):
                st.write(final_response_content)

            st.session_state.messages.append({"role": "assistant", "content": final_response_content})
            
        except Exception as e:
            error_message = f"An error occurred during agent execution: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})