from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults, TavilyAnswer
import os
from AQI_PAI import get_AQI
from langchain_core.tools import Tool
from langchain_ollama import OllamaLLM, ChatOllama
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory
from typing import Optional
os.environ["TAVILY_API_KEY"] = "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"

# Load the BLIP processor and model
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, blip_model = load_blip_model()

def caption_image(image_bytes):
    try:
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return f"Error opening or processing image: {e}"

    inputs = processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

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

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    image_caption: Optional[str] = None
    uploaded_image: Optional[bytes] = None

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
        self.memory = ConversationBufferMemory(memory_key="messages", return_messages=True)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

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

prompt = """You are a smart research assistant. You can understand image descriptions and search the web. \
Use the search engine to look up information. You are allowed to make multiple calls. \
Only look up information when you are sure of what you want. \
If you have processed an image, refer to its description for follow-up questions.
"""

# Initialize the LangGraph agent
abot = Agent(tools, system=prompt)

def run_conversation(user_input: str, image_path: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None):
    messages = []
    if history:
        for item in history:
            if item["role"] == "user":
                messages.append(HumanMessage(content=item["content"]))
            elif item["role"] == "assistant":
                messages.append(AIMessage(content=item["content"]))

    if image_path:
        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            caption = caption_image(image_bytes)
            print(f"Image Caption: {caption}")
            user_prompt_with_image = f"The user uploaded an image with the caption: '{caption}'. Respond to this."
            messages.append(HumanMessage(content=user_prompt_with_image))
            result = abot.graph.invoke({"messages": messages, "image_caption": caption, "uploaded_image": image_bytes})
            response = result['messages'][-1].content
            print(f"Assistant: {response}")
            return {"role": "assistant", "content": response}, [{"role": "user", "content": user_prompt_with_image}, {"role": "assistant", "content": response}]
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            messages.append(HumanMessage(content=user_input))
            result = abot.graph.invoke({"messages": messages})
            response = result['messages'][-1].content
            print(f"Assistant: {response}")
            return {"role": "assistant", "content": response}, [{"role": "user", "content": user_input}, {"role": "assistant", "content": response}]
    else:
        messages.append(HumanMessage(content=user_input))
        result = abot.graph.invoke({"messages": messages})
        response = result['messages'][-1].content
        print(f"Assistant: {response}")
        return {"role": "assistant", "content": response}, [{"role": "user", "content": user_input}, {"role": "assistant", "content": response}]

if __name__ == "__main__":
    chat_history = []
    while True:
        user_text = input("You: ")
        if user_text.lower() == "exit":
            break
        image_file = input("Enter image path (optional, or 'no'): ")
        image_path = image_file if image_file.lower() != 'no' else None
        assistant_response, chat_history_update = run_conversation(user_text, image_path, chat_history)
        chat_history.extend(chat_history_update)