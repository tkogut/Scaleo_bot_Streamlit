import requests

# URL to your local Ollama server
ollama_url = "http://localhost:11434/api/generate"

# Example question
input_text = "Podaj stolicę Polski."

# Prepare the payload for the API request
payload = {
    "model": "llama3.2:1b",  # Use an available model
    "prompt": input_text,
    "stream": False
}

# Send the request to the Ollama server
response = requests.post(ollama_url, json=payload)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response
    response_data = response.json()
    answer = response_data.get("response", "No response received.")
    print("Pytanie:", input_text)
    print("Odpowiedź:", answer)
else:
    print("Failed to get a response from the Ollama server. Status code:", response.status_code)
    print("Response content:", response.text)