import sys
import os

# Add the 'src' directory to the system path so Python can find your package
base_path = os.path.dirname(__file__)
sys.path.append(os.path.join(base_path, "src"))

# Now perform the direct import
from langraph_agenticai.main import load_langgraph_agenticai_app

if __name__ == "__main__":
    load_langgraph_agenticai_app()