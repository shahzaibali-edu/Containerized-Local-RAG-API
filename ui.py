import gradio as gr
import requests

def ask_local_ai(message, history):
    url = "http://127.0.0.1:8000/api/v1/chat"
    payload = {"query": message}
    
    try:
        # Send the question to your Docker backend
        response = requests.post(url, json=payload)
        data = response.json()
        
        # Extract the answer from the JSON dictionary
        return data.get("response", "No response found in API.")
        
    except requests.exceptions.ConnectionError:
        return "⚠️ Error: Cannot connect to the API. Is Docker running?"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Build the chat interface
demo = gr.ChatInterface(
    fn=ask_local_ai,
    title="Internship RAG Pipeline",
    description="Running locally via Docker and SmolLM2"
)

if __name__ == "__main__":
    demo.launch()