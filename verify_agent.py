from langgraph.prebuilt import create_react_agent
import inspect

try:
    with open("sig.txt", "w") as f:
        f.write(f"Version: 1.0.3\n")
        sig = inspect.signature(create_react_agent)
        f.write(f"Signature: {sig}\n")
        f.write(f"Docstring: {create_react_agent.__doc__}\n")
except Exception as e:
    with open("sig.txt", "w") as f:
        f.write(f"Error: {e}\n")
