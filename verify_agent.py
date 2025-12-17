from langgraph.prebuilt import create_react_agent
import inspect

try:
    sig = inspect.signature(create_react_agent)
    print("Signature:", sig)
except Exception as e:
    print(e)
