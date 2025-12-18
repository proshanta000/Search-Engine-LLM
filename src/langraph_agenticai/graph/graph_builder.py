from langgraph.graph import StateGraph, START, END
from src.langraph_agenticai.state.state import State
from langgraph.prebuilt import tools_condition, ToolNode

from src.langraph_agenticai.tools.search_tool import get_tools, create_tool_node
from src.langraph_agenticai.nodes.chatbot_with_tools_node import ChatbotWithToolNode


class GraphBuilder:
    """
    Class to build and manage the LangGraph state graph.
    """
    def __init__(self, model):
        """
        Initialize the GraphBuilder with a specific LLM model.
        
        Args:
            model: The LLM model instance to be used by the graph nodes.
        """
        self.llm = model
        self.graph_builder=StateGraph(State)



    def chatbot_with_tools_building_graph(self):
        """
        Build an advanced chatbot graph with tool integration.
        This method creates a chatbot graph that includes both a chatbot node
        and a tool node. it defines tools, initializes the chatbot with tool
        capabilities, and sets up conditional and direct edges between nodes.
        The chatbot node is set the entry point.
        """

        # Define the tool and tool node
        tools = get_tools()
        tools_node = create_tool_node(tools)

        # Define the llm 
        llm = self.llm

        # Define the chatbot node
        obj_chatbot_node = ChatbotWithToolNode(llm)
        chatbot_node = obj_chatbot_node.create_chatbot(tools)

        # --- node ---
        # Add the chatbot node to the graph
        self.graph_builder.add_node("chatbot",chatbot_node )
        self.graph_builder.add_node("tools", tools_node)

        # --- edge ---
        # Define the flow: Start -> Chatbot -><- tools -> End
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
            {"tools": "tools", END: END}
        )
        self.graph_builder.add_edge("tools", "chatbot")



    def setup_graph(self):
        """
        Sets up the graph for the selected use case.
        """

        
        self.chatbot_with_tools_building_graph()

        
        return self.graph_builder.compile()

        