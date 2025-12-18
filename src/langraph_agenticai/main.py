import streamlit as st
from src.langraph_agenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langraph_agenticai.LLMS.groqllm import GroqLLM
from src.langraph_agenticai.LLMS.geminillm import GeminiLLM
from src.langraph_agenticai.graph.graph_builder import GraphBuilder
from src.langraph_agenticai.ui.streamlitui.display_result import DisplayResultStreamlit

@st.cache_resource
def get_graph(_model):
    """Caches the compiled graph so it doesn't rebuild on every rerun."""
    builder = GraphBuilder(model=_model)
    return builder.setup_graph()

def load_langgraph_agenticai_app():
    ui = LoadStreamlitUI()
    config_input = ui.load_streamlit_ui()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Get the user input
    prompt = st.chat_input("How can I help you today?")
    
    # Initialize the display handler
    display = DisplayResultStreamlit(None, prompt)
    
    if prompt:
        try:
            # 1. Setup LLM
            if config_input["selected_llm"] == "Groq":
                llm = GroqLLM(user_controls_input=config_input).get_llm_model()
            else:
                llm = GeminiLLM(user_controls_input=config_input).get_llm_model()
            
            # 2. Setup Graph
            graph = get_graph(llm)
            display.graph = graph 
            
            # 3. Process and Render
            display.display_result_on_ui()
            
        except Exception as e:
            st.error(f"Application Error: {e}")
    else:
        # Just show history if the user hasn't typed anything new
        display.display_result_on_ui()