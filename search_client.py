import requests

api_url = "http://localhost:8000"

# Тест NLP-пайплайна
text_to_process = "Это тестовый текст для обработки нашим NLP-пайплайном."
response = requests.post(f"{api_url}/process_text", json={"text": text_to_process})
print("Processed tokens:", response.json()["processed_tokens"])

# Тест поиска
search_query = "Хороший товар"
response = requests.post(f"{api_url}/search", json={"text": search_query})
print("Search results:", response.json()["results"])