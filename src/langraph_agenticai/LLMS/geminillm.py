import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

class GeminiLLM:
    """
    Handles initialization and configuration of the Gemini LLM.
    """
    def __init__(self, user_controls_input):
        """
        Initialize with user input controls.
        
        Args:
            user_controls_input (dict): Dictionary containing API keys and model selections.
        """
        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        """
        Retrieves the configured Gemini LLM instance.

        Returns:
            ChatGoogleGenerativeAI: An instance of the Gemini model.
        
        Raises:
            ValueError: If there is an error in initializing the model.
        """
        try:
            gemini_api_key = self.user_controls_input["GEMINI_API_KEY"]
            selected_gemini_model = self.user_controls_input["selected_gemini_model"]

            if gemini_api_key=='' and os.environ["GEMINI_API_KEY"] == '':
                st.error("Please Enter the Gemini API Key")

            llm = ChatGoogleGenerativeAI(api_key=gemini_api_key, model=selected_gemini_model)

        except Exception as e:
            raise ValueError(f"Error Occurred with Exception: {e}")
        
        return llm