import os
import importlib

def explore_module(module_name):
    try:
        module = importlib.import_module(module_name)
        module_path = module.__path__[0]
        print(f"Zawartość modułu '{module_name}':")
        print(dir(module))
        print(f"Pliki w katalogu '{module_path}':")
        print(os.listdir(module_path))
    except ImportError as e:
        print(f"Błąd importu: {e}")

# List of submodules to explore
submodules = [
    'llama_index.embeddings.openai'
]

for submodule in submodules:
    explore_module(submodule)