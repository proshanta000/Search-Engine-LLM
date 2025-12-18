import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import uuid 
from langchain_core.tools import Tool 
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
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

search_wrapper = DuckDuckGoSearchAPIWrapper() 
search = Tool.from_function(
    func=search_wrapper.run,
    name="Search",
    description="A tool for performing general web searches using DuckDuckGo."
)

ALL_TOOLS = [search, arxiv, wiki]
SYSTEM_PROMPT = "You are a helpful assistant..."

# --- 2. Streamlit UI Setup ---
st.title("🗨️Chat with 🔎 Search (Groq Agent)") # Removed extra colon
st.caption("A multi-agent framework powered by Groq and Gemini.")

with st.sidebar: # Added missing colon
    st.markdown("## ⚙️ Configuration Settings")
    
    llm_options = ["Groq", "Gemini"]
    llm_provider = st.selectbox("Select LLM Provider", llm_options)

    if llm_provider == "Groq":
        st.markdown("### 🤖 Groq Settings")
        # Rename this variable to avoid clashing with the actual model object later
        selected_model = st.selectbox("Select Groq Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
        api_key = st.text_input("Enter your Groq API Key:", type="password")
    else:
        st.markdown("### 📝 Gemini Settings")
        selected_model = st.selectbox("Select Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
        api_key = st.text_input("Enter your Gemini API Key:", type="password")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state["messages"] = [{"role": "assistant", "content": "History cleared! How can I help?"}]
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.rerun()

# --- 3. Session State Initialization ---

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you? "}
    ]

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

# Logic to reset agent if user changes Provider or Model
if "current_config" not in st.session_state:
    st.session_state["current_config"] = {"provider": llm_provider, "model": selected_model}

if (st.session_state["current_config"]["provider"] != llm_provider or 
    st.session_state["current_config"]["model"] != selected_model):
    st.session_state["agent_executor"] = None  # Force re-initialization
    st.session_state["current_config"] = {"provider": llm_provider, "model": selected_model}

# Initialize agent_executor
if "agent_executor" not in st.session_state and api_key:
    try:
        if llm_provider == "Groq":
            llm_obj = ChatGroq(groq_api_key=api_key, model=selected_model, streaming=True)
        elif llm_provider == "Gemini":
            llm_obj = ChatGoogleGenerativeAI(api_key=api_key, model=selected_model, streaming=True)

        memory = MemorySaver()
        st.session_state["agent_executor"] = create_agent(
            model=llm_obj, 
            tools=ALL_TOOLS,
            system_prompt=SYSTEM_PROMPT, 
            checkpointer=memory,
        )
    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.session_state["agent_executor"] = None

# Display existing messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 4. Handle User Input ---

if prompt := st.chat_input(placeholder="what is machine learning?"):
    if not api_key or st.session_state.get("agent_executor") is None:
        st.warning("Please provide an API Key.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    agent_executor = st.session_state["agent_executor"]

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            try:
                langchain_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))

                response = agent_executor.invoke(
                    {"messages": langchain_messages},
                    config={"configurable": {"thread_id": st.session_state["thread_id"]}}      
                )

                

                # Extract the final answer from the messages list
                final_response_content = response["messages"][-1].content

                # 2. Display final response and update session state
                response_placeholder.markdown(final_response_content)
                st.session_state.messages.append({'role':"assistant", "content": final_response_content})

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                # Don't forget to save the error to state if you want it to persist
                st.session_state.messages.append({"role": "assistant", "content": error_message})