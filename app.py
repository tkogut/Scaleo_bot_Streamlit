import streamlit as st
import os
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.llms import CustomLLM
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from types import SimpleNamespace  # Dodano SimpleNamespace
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Dodaj kontekst uruchomieniowy Streamlit
add_script_run_ctx()

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfiguracja Streamlit
st.set_page_config(page_title="Chatbot z własnymi danymi", page_icon="🤖")

# Ścieżka do folderu z danymi
data_path = "C:/Users/tkogut/.vscode/Scaleo_bot_Streamlit/dane"

# Ustawienia dla LlamaIndex
Settings.llm = None  # Wyłącz domyślny model językowy (OpenAI)

# Definicja modelu embeddingowego
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Klasa do integracji Ollamy
class OllamaLLM(CustomLLM, BaseModel):
    ollama_url: str = Field(..., description="URL do instancji Ollamy")
    model: str = Field(..., description="Nazwa modelu Ollamy")

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        import requests
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
        if response.status_code == 200:
            response_data = response.json()
            return CompletionResponse(text=response_data.get("response", ""))
        else:
            raise Exception(f"Błąd podczas generowania odpowiedzi: {response.status_code}")

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        import requests
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        response = requests.post(f"{self.ollama_url}/api/generate", json=payload, stream=True)
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    yield CompletionResponse(text=chunk.decode("utf-8"))
        else:
            raise Exception(f"Błąd podczas generowania odpowiedzi: {response.status_code}")

    @property
    def metadata(self) -> SimpleNamespace:  # Zwracamy obiekt SimpleNamespace zamiast słownika
        return SimpleNamespace(
            model_name=self.model,
            ollama_url=self.ollama_url,
            context_window=4096,  # Ustaw odpowiednią wartość dla modelu
            num_output=512  # Ustaw odpowiednią wartość dla modelu
        )

# Konfiguracja Ollamy
ollama_url = "http://localhost:11434"
ollama_model = "llama3.2:1b"  # Użyj modelu, który masz dostępny
llm = OllamaLLM(ollama_url=ollama_url, model=ollama_model)

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
            logger.info(f"Wczytano dokument z {file_path}: {content[:100]}...")  # Log pierwszych 100 znaków

    # Tworzenie indeksu z użyciem modelu embeddingowego
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    logger.info("Indeks LlamaIndex załadowany pomyślnie.")
    return index

# Wczytanie danych i utworzenie indeksu
index = load_data()

# Przykładowe narzędzia
def search_tool(input: Dict[str, Any]) -> Dict[str, Any]:
    """Narzędzie do wyszukiwania informacji w indeksie."""
    try:
        query = input.get("input", "")  # Pobierz zapytanie z klucza 'input'
        results = index.query(query)
        # Zwróć odpowiedź w formacie JSON
        return {
            "thought": "Użyłem narzędzia search_tool, aby znaleźć odpowiedź.",
            "action": "search_tool",
            "input": str(results.response)  # Konwersja na string
        }
    except Exception as e:
        logger.error(f"Błąd podczas wyszukiwania: {e}")
        return {
            "thought": "Wystąpił błąd podczas wyszukiwania.",
            "action": "error",
            "input": "Przepraszam, wystąpił błąd podczas wyszukiwania."
        }

def summarize_tool(input: Dict[str, Any]) -> Dict[str, Any]:
    """Narzędzie do podsumowywania tekstu."""
    try:
        text = input.get("input", "")  # Pobierz tekst z klucza 'input'
        # Tutaj możesz dodać logikę podsumowywania
        return {
            "thought": "Użyłem narzędzia summarize_tool, aby podsumować tekst.",
            "action": "summarize_tool",
            "input": f"Podsumowanie: {text[:100]}..."  # Przykładowe podsumowanie
        }
    except Exception as e:
        logger.error(f"Błąd podczas podsumowywania: {e}")
        return {
            "thought": "Wystąpił błąd podczas podsumowywania.",
            "action": "error",
            "input": "Przepraszam, wystąpił błąd podczas podsumowywania."
        }

# Tworzenie narzędzi
search_tool = FunctionTool.from_defaults(fn=search_tool)
summarize_tool = FunctionTool.from_defaults(fn=summarize_tool)

# Konfiguracja agenta
agent = ReActAgent.from_tools(
    tools=[search_tool, summarize_tool],  # Dodaj narzędzia tutaj
    llm=llm,
    verbose=True,
    max_iterations=5  # Ogranicz liczbę iteracji, aby uniknąć nieskończonych pętli
)

# Funkcja do generowania odpowiedzi
def generate_response(prompt: str) -> str:
    """Generate a response using the agent."""
    try:
        response = agent.chat(prompt)
        return response.response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Przepraszam, wystąpił błąd podczas generowania odpowiedzi. Spróbuj ponownie."

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
        # Generate response using the agent
        response = generate_response(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})