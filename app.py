import streamlit as st
from llama_index.core import VectorStoreIndex, Document, ServiceContext, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInpitPrompt
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfiguracja Streamlit
st.set_page_config(page_title="Chatbot z własnymi danymi", page_icon="🤖")

# Ścieżka do lokalnego modelu Llama
model_path = "C:/Users/tkogut/.vscode/models/llama-3.2-1B-local"

# Inicjalizacja tokenizera i modelu
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Inicjalizacja modelu HuggingFace
llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
)

# Ścieżka do folderu z danymi
data_path = "C:/Users/tkogut/.vscode/Scaleo_bot_Streamlit/dane"

# Funkcja do wczytywania dokumentów
@st.cache_resource
def load_data():
    # Sprawdzenie folderu 'dane'
    if not os.path.exists(data_path):
        raise ValueError(f"Folder {data_path} nie istnieje.")

    files = os.listdir(data_path)
    if not files:
        raise ValueError(f"Brak plików w folderze {data_path}.")

    logger.info(f"Zawartość folderu '{data_path}': {files}")

    # Wczytanie dokumentów
    documents = []
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append(Document(text=content))
            logger.info(f"Wczytano dokument z {file_path}")

    # Tworzenie indeksu
    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    logger.info("Indeks LlamaIndex załadowany pomyślnie.")
    return index

index = load_data()

# Inicjalizacja chat engine
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Interfejs użytkownika
st.title("Chatbot z własnymi danymi 🤖")
st.write("Witaj! Jestem chatbotem, który może odpowiedzieć na pytania na podstawie dostarczonych danych.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Jakie masz pytanie?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chat_engine.chat(prompt)
        st.markdown(response.response)
        st.session_state.messages.append({"role": "assistant", "content": response.response})