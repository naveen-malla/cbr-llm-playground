import os
import streamlit as st

st.set_page_config(page_title="CBR Lab (Local)", page_icon="🧠")
st.title("CBR Lab (Local) — CBR + Local LLM via Ollama")

st.markdown("""
**What is this?**  
A hands-on mini-lab to learn Case-Based Reasoning (CBR) and how a local LLM can help with adaptation & explanations.

**Pages** (left sidebar):
1. Inspect Cases
2. Retrieve Similar
3. LLM Adaptation
4. Hybrid RAG

**Local LLM**: Uses [Ollama](https://ollama.com). Default model: `phi3:mini` (edit `.env` to change).
""")

st.info(f"Ollama model: `{os.environ.get('OLLAMA_MODEL','phi3:mini')}`. Make sure you ran `ollama pull {os.environ.get('OLLAMA_MODEL','phi3:mini')}`.")
