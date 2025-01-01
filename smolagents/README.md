## Huggingface smolagents examples
Examples leveraging Huggingface smolagents library

### Features
Reference: https://huggingface.co/blog/smolagents
smolagents is the successor to transformers.agents, and will be replacing it as transformers.agents gets deprecated in the future.

### Getting Started
1. Install required libraries
```bash
pip install -r requirements.txt
```
2. Ensure ollama is installed and llama3.2 model available
```bash
check_ollama_model_exists llama3.2
```

3. Run the examples
```bash
streamlit run smolagents_ex1.py
```

### Base tools loaded by default:
DuckDuckGo web search*: performs a web search using DuckDuckGo browser.
Python code interpreter: runs your the LLM generated Python code in a secure environment.
Transcriber: a speech-to-text pipeline built on Whisper-Turbo that transcribes an audio to text.

### References:
https://huggingface.co/spaces/vincentiusyoshuac/iChat/blob/main/app.py

https://huggingface.co/blog/smolagents

https://github.com/huggingface/smolagents/tree/main


