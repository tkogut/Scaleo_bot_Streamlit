import os
import json
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.embeddings.loading import load_embed_model
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.readers import Document
from sentence_transformers import SentenceTransformer

def test_llama_index():
    # Ustaw model embeddings na MockEmbedding
    embed_model = load_embed_model({
        "class_name": "MockEmbedding",  # Replace with the correct class name if needed
        "model_name": "dkleczek/bert-base-polish-cased-v1",
        "embed_dim": 768  # Add the embed_dim argument
    })

    # Ustaw model embeddings w Settings
    from llama_index.core import Settings
    Settings.embed_model = embed_model
    Settings.llm = None  # Disable the default LLM

    # Create a sample document
    documents = [Document(text="To jest przykładowy tekst do testowania modelu.")]

    # Create the test_indexes directory if it does not exist
    persist_dir = "test_indexes"
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    # Create index
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir=persist_dir)
    print("Index created and persisted successfully.")

    # Verify the contents of the persist directory
    print(f"Contents of '{persist_dir}' directory: {os.listdir(persist_dir)}")

    # Verify the contents of the docstore.json file
    docstore_path = os.path.join(persist_dir, "docstore.json")
    if os.path.exists(docstore_path):
        with open(docstore_path, 'r', encoding='utf-8') as f:
            docstore_content = f.read()
            print(f"Contents of 'docstore.json': {docstore_content}")
    else:
        print(f"File 'docstore.json' does not exist in '{persist_dir}' directory.")

    # Load the index and perform a query
    loaded_storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    loaded_index = load_index_from_storage(loaded_storage_context)
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query("Co zawiera przykładowy plik tekstowy?")
    print("Odpowiedź:", response.response)

if __name__ == "__main__":
    test_llama_index()