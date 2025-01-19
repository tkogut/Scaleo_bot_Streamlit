try:
    from llama_index.core import VectorStoreIndex, Document
    from llama_index.llms.huggingface import HuggingFaceLLM
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    print("Importy działają poprawnie.")
except ImportError as e:
    print(f"Błąd importu: {e}")