# 🧠 Multimodal Research Assistant – AutoMates

This project is a final-year research assistant platform built using **LangGraph**, **Streamlit**, **BLIP**, and **ChatOllama (Mistral-Nemo)**. It supports multimodal inputs, hybrid similarity search, ticket management, and integrates various APIs like **Tavily**, **OpenMeteo**, and **OpenAI** for enhanced research and automation.

---

## 📦 Features

- 📄 Text and 🖼️ image input support via BLIP
- 🔁 LangGraph-based step-wise flow control
- 🌐 Tavily-powered web search and contextual enrichment
- 🌦️ AQI and weather information via OpenMeteo
- 🧠 Backed by Mistral-Nemo via ChatOllama for LLM responses

---

## 🚀 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Pai1403/Single-Agent_LLM
cd Single-Agent_LLM
```

2.Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

3.Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 API Key Setup

This project uses the following APIs:
- OpenAI
- Tavily
- OpenMeteo (no key needed)

1. Create a .env file in the root directory:

``` bash
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
BASE_URL=http://localhost:11434/v1  # For ChatOllama with Mistral-Nemo
```
save it in .env file

---
#🧪 Running the App

Run the Streamlit interface:

```bash
streamlit run app.py
```

---
# Model Setup (Mistral-Nemo via ChatOllama)

If you're using Ollama to run Mistral locally:
1. Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. Pull the Mistral-Nemo model:

```bash
ollama pull mistral:nemo
```

3. Start the model:

```bash
ollama run mistral:nemo
```
Ensure your .env file uses the correct BASE_URL.

---
#  Image Captioning (BLIP)

This project assumes BLIP is integrated for image-to-text conversion.
If you're not using a prepackaged version, install directly from GitHub:

```bash
pip install git+https://github.com/salesforce/BLIP.git
```

---
#  Sample Use Case
- Upload an image or enter a query.
- The assistant performs multimodal analysis using BLIP and LangGraph pipeline.
- Get summaries powered by Mistral or OpenAI.

#  Troubleshooting
Missing keys: Ensure .env is correctly configured and loaded.
Ollama errors: Make sure the model is running and BASE_URL is correct.
Dependencies: If modules are missing, re-run ```pip install -r requirements.txt.```


#📄 License
MIT License – Feel free to use and modify this project for educational purposes.
