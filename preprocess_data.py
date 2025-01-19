import os
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # Poprawiony import
from llama_index.core import Settings

def preprocess_data():
    # Ustaw model embeddings na HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    # Ustaw model embeddings w Settings
    Settings.embed_model = embed_model
    Settings.llm = None  # Wyłącz domyślny model językowy

    # Sprawdź zawartość folderu 'dane'
    input_dir = "dane"
    if not os.path.exists(input_dir):
        raise ValueError(f"Folder {input_dir} nie istnieje.")
    
    files = os.listdir(input_dir)
    if not files:
        raise ValueError(f"Brak plików w folderze {input_dir}.")
    
    print(f"Zawartość folderu '{input_dir}': {files}")
    
    # Ręczne wczytanie plików i utworzenie obiektów Document
    documents = []
    for file in files:
        file_path = os.path.join(input_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append(Document(text=content))  # Użyj Document z schema
            print(f"Wczytano dokument z {file_path}")

    # Tworzenie indeksu
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    
    # Zapisz indeks do folderu 'indexes'
    index.storage_context.persist(persist_dir="indexes")
    print("Indeks został utworzony i zapisany pomyślnie.")

if __name__ == "__main__":
    preprocess_data()