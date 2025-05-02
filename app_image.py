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
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

os.environ["TAVILY_API_KEY"] = "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"

# Load the BLIP processor and model
@st.cache_resource
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

prompt = """You are a smart research assistant. You can understand image descriptions and search the web. \
Use the search engine to look up information. You are allowed to make multiple calls. \
Only look up information when you are sure of what you want. \
If you have processed an image, refer to its description for follow-up questions.
"""

# Initialize the LangGraph agent
abot = Agent(tools, system=prompt)

st.title("LangGraph Research Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "image_caption" not in st.session_state:
    st.session_state["image_caption"] = None
if "image_llm_response" not in st.session_state:
    st.session_state["image_llm_response"] = None
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user" and "image" in msg:
        st.chat_message(msg["role"]).image(msg["image"], caption="Uploaded Image")
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    st.session_state["uploaded_image"] = image
    st.image(st.session_state["uploaded_image"], caption="Uploaded Image", use_column_width=True)
    caption = caption_image(image_bytes)
    st.session_state["image_caption"] = caption
    st.write(f"Image Caption: {st.session_state['image_caption']}")
    user_prompt_with_image = f"The user uploaded an image with the caption: '{st.session_state['image_caption']}'. Respond to this."
    st.session_state["messages"].append({"role": "user", "content": user_prompt_with_image, "image": st.session_state["uploaded_image"]})

    with st.chat_message("assistant"):
        st.write("Thinking about the image...")
        messages = [HumanMessage(content=user_prompt_with_image)]
        result = abot.graph.invoke({"messages": messages})
        response = result['messages'][-1].content
        st.write(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.session_state["image_llm_response"] = response

# Chat input for text queries
if text_prompt := st.chat_input("Ask me anything (or about the image above):"):
    st.session_state["messages"].append({"role": "user", "content": text_prompt})
    st.chat_message("user").write(text_prompt)

    # If there was an image uploaded and an initial LLM response, we can refer to it
    if st.session_state["image_llm_response"]:
        contextualized_prompt = f"Referring to the previous image and its description, as well as the AI's response: '{st.session_state['image_llm_response']}', answer this question: {text_prompt}"
        messages = [HumanMessage(content=contextualized_prompt)]
    else:
        messages = [HumanMessage(content=text_prompt)]

    with st.chat_message("assistant"):
        st.write("Thinking...")
        result = abot.graph.invoke({"messages": messages})
        response = result['messages'][-1].content
        st.write(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})

# Reset button
if st.button("Reset Conversation"):
    st.session_state["messages"] = []
    st.session_state["image_caption"] = None
    st.session_state["image_llm_response"] = None
    st.session_state["uploaded_image"] = None
    uploaded_file = None  # Reset the uploaded_file variable
    st.session_state["file_uploader"] = None # Clear the file uploader's state
    st.rerun()