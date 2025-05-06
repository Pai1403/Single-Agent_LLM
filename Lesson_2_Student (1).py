
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults,TavilyAnswer
import os
from AQI_PAI import get_AQI
from langchain_core.tools import Tool
from langchain_ollama import OllamaLLM
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama

os.environ["TAVILY_API_KEY"] = "API_KEY"
##########################################################################################
# class TavilySearch:
#     name = "search_web"  # custom name

#     def invoke(self, args: dict) -> str:
#         query = args.get("query")
#         if not query:
#             return "Missing 'query' in arguments."
        
#         tool = TavilySearchResults(max_results=5)
#         return tool.invoke({"query": query})
    
def search_web(query: str) -> str:
    # tool = TavilySearchResults(max_results=5)
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
#############TOOLS######################
# aqi_tool = AQITool()
# web_search=TavilyAnswer(max_results=4)
# tools = [TavilySearch(), aqi_tool]
# tool = TavilySearchResults(max_results=4) #increased number of results
tools = [ aqi_tool, search_web_tool]


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

    def __init__(self,  tools, system=""):  # def __init__(self,model, tools, system=""): - when running on proxim
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
        self.model = ChatOllama(base_url='http://localhost:11434', model='mistral-nemo') #qwen3:14b
        # self.model = model.bind_tools(tool_schemas)     #uncomment it while running on proxim
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
    # def call_openai(self, state: AgentState):
    #     messages = state['messages']
    #     if self.system:
    #         messages = [SystemMessage(content=self.system)] + messages

    #     # 🧠 Limit number of past messages (keep only last 4-6 + system)
    #     if len(messages) > 7:
    #         messages = [messages[0]] + messages[-6:]

    #     message = self.model.invoke(messages)
    #     return {'messages': [message]}

    
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



    # def take_action(self, state: AgentState):
    #     tool_calls = state['messages'][-1].tool_calls
    #     results = []
    #     for t in tool_calls:
    #         print(f"Calling: {t}")
    #         if not t['name'] in self.tools:      # check for bad tool name from LLM
    #             print("\n ....bad tool name....")
    #             result = "bad tool name, retry"  # instruct LLM to retry if bad
    #         else:
    #             result = self.tools[t['name']].invoke(t['args'])
    #         results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    #     print("Back to the model!")
    #     return {'messages': results}

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
from openai import OpenAI  # or your specific SDK if this is not the OpenAI package

# Constants
# MODEL = 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'
# API_KEY = <API_Key> 
# BASE_URL = <Base_user>

# # Initialize client
# client = OpenAI(api_key=API_KEY)
# client.base_url = BASE_URL 


# model = ChatOpenAI(
#     model_name=MODEL,
#     openai_api_key=API_KEY,
#     base_url=BASE_URL,
#     max_tokens=2000)

# Results may vary per run and over time as search information and models change.
#-------------------------
query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
What is the GDP of that state? Answer each question.?" 
messages = [HumanMessage(content=query)]

# model = ChatOpenAI(model="gpt-4o")  # requires more advanced model
# abot = Agent(tools, system=prompt) # 
abot = Agent(tools, system=prompt) #- when running on proxim
result = abot.graph.invoke({"messages": messages})


print(result['messages'][-1].content)

