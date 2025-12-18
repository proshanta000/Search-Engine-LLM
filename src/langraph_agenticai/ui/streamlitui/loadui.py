import os
import streamlit as st
from src.langraph_agenticai.ui.uiconfigfile import Config

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):
        # 1. Page Configuration
        try:
            st.set_page_config(page_title="🤖 " + self.config.get_page_title(), layout="wide")
        except st.errors.StreamlitAPIException:
            pass
        
        st.header("🤖 " + self.config.get_page_title())

        with st.sidebar:
            st.markdown("## ⚙️ Configuration Settings")
            llm_options = self.config.get_llm_options()

            # 2. LLM Selection
            st.markdown("### 🤖 LLM Selection")
            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options)

            # 3. Dynamic Model & Key Handling
            if self.user_controls["selected_llm"] == "Groq":
                model_options = self.config.get_groq_model_options()
                self.user_controls["selected_groq_model"] = st.selectbox("Select Model", model_options)
                
                # Persistence: Check if key exists in session state to keep it in the text box
                saved_groq_key = st.session_state.get("GROQ_API_KEY", "")
                groq_key = st.text_input("Groq API Key", type="password", value=saved_groq_key)
                
                self.user_controls["GROQ_API_KEY"] = groq_key
                st.session_state["GROQ_API_KEY"] = groq_key

                if not groq_key:
                    st.warning("⚠️ Enter GROQ API Key: [Get it here](https://console.groq.com/keys)")
            
            elif self.user_controls["selected_llm"] == "Gemini":
                model_options = self.config.get_gemini_model_options()
                self.user_controls["selected_gemini_model"] = st.selectbox("Select Model", model_options)
                
                saved_gemini_key = st.session_state.get("GEMINI_API_KEY", "")
                gemini_key = st.text_input("Gemini API Key", type="password", value=saved_gemini_key)
                
                self.user_controls["GEMINI_API_KEY"] = gemini_key
                st.session_state["GEMINI_API_KEY"] = gemini_key

                if not gemini_key:
                    st.warning("⚠️ Enter Gemini API Key: [Get it here](https://aistudio.google.com/api-keys)")

            st.divider()
            
            # 4. Action Buttons
            if st.button("🗑️ Clear Chat History"):
                if "chat_history" in st.session_state:
                    st.session_state.chat_history = []
                st.rerun()

        return self.user_controls