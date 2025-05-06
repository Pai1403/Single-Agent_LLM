import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults, TavilyAnswer
import os
from AQI_PAI import get_AQI
from langchain_core.tools import Tool
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama

os.environ["TAVILY_API_KEY"] = "API_KEY"

def search_web(query: str) -> str:
    tool = TavilyAnswer(max_results=5)
    return tool.invoke({"query": query})

search_web_tool = Tool.from_function(
    func=search_web,
    name="search_web",
    description="Search the web for a given query and return top relevant results.",
)

aqi_tool = Tool.from_function(
    func=get_AQI,
    name="get_AQI",
    description="Get the air quality data for a specific city or location (e.g., Delhi)."
)

tools = [aqi_tool, search_web_tool]

tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "get_AQI",
            "description": "Get the air quality data for a specific location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "place_name": {
                        "type": "string",
                        "description": "Name of the city or location, eg: Delhi"
                    }
                },
                "required": ["place_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Perform a web search and return top relevant results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to look up online"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = ChatOllama(base_url='http://localhost:11434', model='mistral-nemo')
        self.model = self.model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []

        for t in tool_calls:
            print(f"Calling tool: {t}")
            tool_name = t['name']
            tool_args = t['args']
            tool_id = t['id']

            if tool_name not in self.tools:
                print(f"[ERROR] Tool '{tool_name}' not found.")
                result = f"Tool '{tool_name}' not recognized."
            else:
                tool = self.tools[tool_name]
                try:
                    if not isinstance(tool_args, dict):
                        print(f"[ERROR] Args for '{tool_name}' are not a dict: {tool_args}")
                        result = f"Invalid arguments for tool '{tool_name}'."
                    else:
                        result = tool.invoke(tool_args)
                        print(f"Result: {result}")
                except Exception as e:
                    print(f"[ERROR] Tool '{tool_name}' failed with error: {e}")
                    result = f"Tool '{tool_name}' failed with error: {e}"

            results.append(ToolMessage(tool_call_id=tool_id, name=tool_name, content=str(result)))

        print("Returning to model.")
        return {'messages': results}

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

# Initialize the LangGraph agent
abot = Agent(tools, system=prompt)

st.title("LangGraph Research Assistant")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Chat input
if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st.write("Thinking...")
        messages = [HumanMessage(content=prompt)]
        result = abot.graph.invoke({"messages": messages})
        response = result['messages'][-1].content
        st.write(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})

# Reset button
if st.button("Reset Conversation"):
    st.session_state["messages"] = []
    st.rerun()
