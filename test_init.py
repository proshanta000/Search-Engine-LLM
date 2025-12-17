from app import initialize_agent_executor
from langchain_core.tools import tool
import os

# Mock tools
@tool
def mock_tool(query: str):
    """A mock tool."""
    return "mock result"

# Patch st.error
import streamlit as st
st.error = lambda x: print(f"Captured Error: {x}")

def test_init():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Skipping test: GROQ_API_KEY not found.")
        return

    try:
        executor = initialize_agent_executor(api_key, [mock_tool])
        if executor:
            print("Agent executor initialized successfully!")
        else:
            print("Agent executor initialization returned None.")
    except Exception as e:
        print(f"Agent executor initialization failed: {e}")

if __name__ == "__main__":
    test_init()
