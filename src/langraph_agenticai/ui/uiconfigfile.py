import os
from configparser import ConfigParser

class Config:
    """
    Configuration helper class to read settings from the INI file.
    """
    def __init__(self, config_file=None):
        """
        Initialize the Config parser and verify the file exists.
        """
        if config_file is None:
            # Dynamically find the path relative to this file
            base_path = os.path.dirname(__file__)
            config_file = os.path.join(base_path, "uiconfigfile.ini")
        
        self.config = ConfigParser()
        
        # Check if file exists before reading
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found at: {config_file}")
            
        files_read = self.config.read(config_file)
        if not files_read:
            raise ValueError(f"Could not read configuration file: {config_file}")
        


    
    def _get_list(self, key):
        """Helper to get comma-separated values as a cleaned list."""
        val = self.config["DEFAULT"].get(key, "")
        return [item.strip() for item in val.split(",") if item.strip()]

    def get_llm_options(self):
        return self._get_list("LLM_OPTIONS")
    
    def get_groq_model_options(self):
        return self._get_list("GROQ_MODEL_OPTIONS")
    
    def get_gemini_model_options(self):
        return self._get_list("GEMINI_MODEL_OPTIONS")

    def get_page_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE", "AI Agent")