try:
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llama_index.core.agent import ReActAgent
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    print("Importy działają poprawnie.")

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

except ImportError as e:
    print(f"Błąd importu: {e}")