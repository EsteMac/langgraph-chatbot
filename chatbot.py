import os
from dotenv import load_dotenv
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Create a MemorySaver checkpointer
memory = MemorySaver()

# Define the state type
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Initialize the web search tool
tool = TavilySearchResults(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=2
)
tools = [tool]

# Initialize the ChatAnthropic model with the API key from environment variables
llm = ChatAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-5-sonnet-20240620"
)

# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)

# Define the function that will process messages
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")

# Set the entry point
graph_builder.add_edge(START, "chatbot")

# Set the exit point
graph_builder.add_edge("chatbot", END)

# Compile the graph with the provided checkpointer
graph = graph_builder.compile(checkpointer=memory)

# Pick a thread to use as the key for this conversation
config = {"configurable": {"thread_id": "1"}}

# Stream the chat
def stream_graph_updates(user_input: str):
    """Helper function to stream the graph's responses"""
    for event in graph.stream(
        {"messages": [("user", user_input)]}, 
        config=config, 
        stream_mode="values"
    ):
        if "messages" in event:
            print("Assistant:", event["messages"][-1].content)

# Interactive chat loop
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
            
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break