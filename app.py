# Import necessary libraries
import os
from dotenv import load_dotenv
from typing import Annotated, Optional, List, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import  HumanMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Error checking for API keys
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in the .env file")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# Step 1: Define Detailed State for Contract Finding
class ContractState(TypedDict):
    messages: Annotated[list, add_messages]
    industry: Optional[str]
    contract_type: Optional[str]
    location: Optional[str]
    budget_range: Optional[str]

# Initialize Graph Builder
graph_builder = StateGraph(ContractState)

# Step 2: Set up LLM and Web Search Tool
# Initialize the LLM with Groq API
llm = ChatGroq(
    model="llama3-70b-8192",  # High-performance model for complex reasoning
    api_key=GROQ_API_KEY,
    temperature=0.5  # Slightly creative but mostly focused
)

# Initialize web search tool
search_tool = TavilySearchResults(
    max_results=5,  # Increased to get more comprehensive results
    api_key=TAVILY_API_KEY
)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools([search_tool])

# Step 3: Input Gathering Node
def gather_contract_details(state: ContractState):
    """
    Gather and clarify contract search details from the user.
    """
    # If details are missing, ask for more information
    if not all([state.get('industry'), state.get('contract_type'), state.get('location')]):
        clarification_prompt = (
            "I'll help you find contract opportunities. "
            "Could you please provide more details:\n"
            "1. What industry are you looking for contracts in? (e.g., IT, Construction, Marketing)\n"
            "2. What type of contract are you seeking? (e.g., Full-time, Part-time, Project-based)\n"
            "3. In which location are you looking for contracts?\n"
            "4. Do you have a specific budget range in mind?"
        )
        return {"messages": [AIMessage(content=clarification_prompt)]}
    
    # If all details are present, proceed to search
    return {"messages": [AIMessage(content="Great! I'll search for contract opportunities based on your details.")]}

# Step 4: Contract Search Node
def safe_search_contracts(state: ContractState) -> List[Dict[str, str]]:
    """
    Safely search for contracts with error handling.
    """
    try:
        # Construct a detailed search query
        search_query = (
            f"Latest {state.get('contract_type', '')} contract opportunities "
            f"in {state.get('industry', '')} industry "
            f"located in {state.get('location', '')} "
            f"with budget range {state.get('budget_range', 'any')}"
        )
        
        # Invoke search tool
        search_results = search_tool.invoke({"query": search_query})
        
        # Safe processing of search results
        processed_results = []
        for result in search_results:
            # Safely extract information with fallbacks
            processed_result = {
                'title': result.get('title', 'Untitled Contract Opportunity'),
                'url': result.get('url', 'No URL available'),
                'content': result.get('content', 'No description available')
            }
            processed_results.append(processed_result)
        
        return processed_results
    
    except Exception as e:
        print(f"Error in contract search: {e}")
        return []

def search_contracts(state: ContractState):
    """
    Process contract search results and create a message
    """
    search_results = safe_search_contracts(state)
    
    if not search_results:
        result_message = AIMessage(
            content="I'm sorry, but I couldn't find any contract opportunities matching your criteria. "
            "Would you like to modify your search parameters?"
        )
    else:
        # Format results into a readable message
        results_text = "\n\n".join([ 
            f"{i+1}. {result['title']}\n"
            f"   URL: {result['url']}\n"
            f"   Description: {result['content'][:200]}..."  # Truncate long descriptions
            for i, result in enumerate(search_results)
        ])
        
        result_message = AIMessage(
            content=f"I've found some contract opportunities matching your criteria:\n\n{results_text}"
        )
    
    return {"messages": [result_message]}

# Step 5: Add Nodes to Graph
graph_builder.add_node("input_gathering", gather_contract_details)
graph_builder.add_node("contract_search", search_contracts)

# Step 6: Define Routing
def route_to_search(state: ContractState):
    """
    Determine whether to gather more details or proceed to search.
    """
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], HumanMessage):
        # Check if critical details are missing
        if not all([state.get('industry'), state.get('contract_type'), state.get('location')]):
            return "input_gathering"
    return "contract_search"

# Step 7: Add Edges
graph_builder.add_conditional_edges(
    "input_gathering", 
    route_to_search, 
    {"input_gathering": "input_gathering", "contract_search": "contract_search"}
)
graph_builder.add_edge("contract_search", END)
graph_builder.add_edge(START, "input_gathering")

# Step 8: Compile the Graph
contract_finder_graph = graph_builder.compile()

# Step 9: Interactive Function
def run_contract_finder():
    print("Welcome to the Contract Finder Agent!")
    print("I'll help you find the best contract opportunities.")
    
    # Initial state to track contract details
    initial_state = {
        "messages": [],
        "industry": None,
        "contract_type": None,
        "location": None,
        "budget_range": None
    }
    
    while True:
        try:
            user_input = input("\nUser: ")
            
            # Exit condition
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye! Hope you found some great contract opportunities.")
                break
            
            # Add user input to state
            initial_state["messages"].append(HumanMessage(content=user_input))
            
            # Stream graph updates
            for event in contract_finder_graph.stream(initial_state):
                for key, value in event.items():
                    if "messages" in value:
                        last_message = value["messages"][-1]
                        
                        # Update state with any extracted details
                        if isinstance(last_message, AIMessage):
                            print(f"\nAssistant: {last_message.content}")
                        
                        # Attempt to extract details from user input if not already set
                        if not initial_state["industry"] and "industry" in user_input.lower():
                            initial_state["industry"] = user_input.split("industry")[-1].strip()
                        if not initial_state["contract_type"] and "contract type" in user_input.lower():
                            initial_state["contract_type"] = user_input.split("contract type")[-1].strip()
                        if not initial_state["location"] and any(loc in user_input.lower() for loc in ["in", "location"]):
                            initial_state["location"] = user_input.split("location")[-1].strip()
                        if not initial_state["budget_range"] and "budget" in user_input.lower():
                            initial_state["budget_range"] = user_input.split("budget")[-1].strip()
            
        except Exception as e:
            print(f"An error occurred: {e}")
            break

# Main execution
if __name__ == "__main__":
    run_contract_finder()
