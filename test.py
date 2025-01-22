try:
    import streamlit as st
    from typing import List
    import os
    import logging
    from llama_index.core.indices.vector_store import VectorStoreIndex  # Poprawny import
    from llama_index.core import Document, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import requests  # Do integracji z Ollama

    # Check available modules in llama_index
    import llama_index
    print("Available modules in llama_index:")
    print(dir(llama_index))

    # Check available modules in llama_index.core
    import llama_index.core
    print("Available modules in llama_index.core:")
    print(dir(llama_index.core))

    # Check available modules in llama_index.core.llms
    import llama_index.core.llms as llms
    print("Available modules in llama_index.core.llms:")
    print(dir(llms))

    print("Importy działają poprawnie.")
except ImportError as e:
    print(f"Błąd importu: {e}")