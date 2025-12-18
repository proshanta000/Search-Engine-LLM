from src.langraph_agenticai.state.state import State
from langchain_core.messages import SystemMessage

class ChatbotWithToolNode:
    """
    Chatbot logic enhanced with tool integration.
    """

    def __init__(self, model):
        self.llm = model

    def create_chatbot(self, tools):
        """
        Returns a chatbot node function bound to tools.
        """
        # 1. Bind tools to the LLM once during initialization
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State) -> dict:
            """
            LangGraph Node: Invokes the LLM with system instructions and current state.
            """
            messages = state["messages"]
            
            # Define the system behavior
            sys_msg = SystemMessage(content="""You are a helpful and versatile AI Assistant. 
        You have access to real-time search tools to provide accurate info.
        1. If the user asks about sports, provide detailed sports analysis.
        2. If the user asks about ANY other topic (news, tech, cooking, etc.), 
           use your tools to find the best information and answer normally.
        3. Do not limit yourself to one persona unless the user asks you to.""")

            # 2. Invoke the LLM
            # Note: We prepend the system message here so the LLM always knows its role,
            # but we only return the NEW response to update the state.
            llm_response = llm_with_tools.invoke([sys_msg] + messages)
            
            # 3. Return the update dictionary
            # The 'messages' key in your State should use a reducer (like add_messages)
            return {"messages": [llm_response]}
        
        return chatbot_node