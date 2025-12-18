from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langgraph.prebuilt import ToolNode
from langchain_community.tools import tool
from langchain_core.tools import Tool 
import arxiv
# Direct implementation of DuckDuckGo search to avoid LangChain wrapper issues
from ddgs import DDGS
from pydantic import BaseModel, Field

def get_tools():
    """
    Return the list of tools to be used in the chatbot.
    """
    
    # 1. Define input schema for tools
    # Renamed: WikipediaInput
    class WikipediaInput(BaseModel):
        query: str = Field(description="Search Query for wikipedia")

    class ArxivInput(BaseModel):
        query: str = Field(description="Search query for Arxiv")

    # 2. Create tools using the @tool decorator.
    @tool("wikipedia_search", args_schema=WikipediaInput)
    def wikipedia_search(query: str) -> str:
        """Search wikipedia for a given query and return the summary."""
        # Instantiate the wrapper each time the tool is called 
        _wikipedia_api_wrapper_instance = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
        return _wikipedia_api_wrapper_instance.run(query)



    @tool("arxiv_search", args_schema=ArxivInput)
    def arxiv_search(query: str) -> str:
        """Search Arxiv for a given query and return the summary."""
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=1,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            results = []
            for result in client.results(search):
                results.append(f"Title: {result.title}\nSummary: {result.summary}\nLink: {result.entry_id}")
            if not results:
                 return "No Arxiv results found."
            return "\n\n".join(results)
        except Exception as e:
            return f"Error performing Arxiv search: {e}"
    

    class SearchInput(BaseModel):
        query: str = Field(description="Search query for general web results")

    @tool("Search", args_schema=SearchInput)
    def search_tool(query: str) -> str:
        """A tool for performing general web searches using DuckDuckGo."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                if not results:
                    return "No results found."
                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append(f"Title: {result['title']}\nLink: {result['href']}\nSnippet: {result['body']}")
                return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error performing search: {e}"

    tools = [wikipedia_search, arxiv_search, search_tool]
    return tools


# Renamed: create_tool_node
def create_tool_node(tools):
    """
    Create and returns a tool node for the graph
    """
    # The ToolNode will automatically handle routing the LLM's requests
    # to the correct tool function in the list.
    return ToolNode(tools=tools)