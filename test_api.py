import requests

url = "http://127.0.0.1:8000/api/v1/chat"
payload = {"query": "What is this document about?"}

print("Asking the AI...")
response = requests.post(url, json=payload)
print(response.json())