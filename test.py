try:
    import streamlit as st
    from llama_index.core import VectorStoreIndex, Document, ServiceContext, SimpleDirectoryReader
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.core.prompts.prompts import SimpleInputPrompt
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import os
    import logging
    print("Importy działają poprawnie.")
except ImportError as e:
    print(f"Błąd importu: {e}")