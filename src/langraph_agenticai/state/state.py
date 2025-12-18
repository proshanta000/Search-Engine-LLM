from typing import Annotated
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    """ 
    Represent the structure of the state used in graph
    
    Attributes:
        messages: A list of messages that represents the conversation history.
                  The `add_messages` reducer handles updates to this list.
    """

    messages: Annotated[list, add_messages]