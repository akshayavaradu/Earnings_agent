import requests
import json

# Ollama API endpoint (assuming Ollama is running locally)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Define a simple tool (function) for adding two numbers
def add_numbers(a: float, b: float) -> float:
    """Adds two numbers and returns the result."""
    return a + b

# Available tools (metadata for the model)
TOOLS = [
    {
        "name": "add_numbers",
        "description": "Adds two numbers and returns the result.",
        "parameters": {
            "a": {"type": "float", "description": "The first number"},
            "b": {"type": "float", "description": "The second number"}
        }
    }
]

# Function to call Ollama API
def call_ollama(prompt, model="llama3.1"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return json.loads(response.text).get("response", "")
    else:
        raise Exception(f"Ollama API error: {response.status_code}")

# Function to process user input and handle tool calling
def process_query(user_input):
    # Create a prompt that includes tool information and user query
    tool_info = json.dumps(TOOLS, indent=2)
    prompt = f"""
You are a helpful assistant with access to the following tools:
{tool_info}

If the user's query requires a tool, respond with a JSON object containing:
- "tool": the name of the tool to call
- "parameters": the parameters for the tool

If no tool is needed, respond with the answer as a plain string.

User query: {user_input}
"""

    # Call the Ollama model
    response = call_ollama(prompt)
    
    try:
        # Try to parse the response as JSON (for tool calls)
        response_json = json.loads(response)
        if "tool" in response_json and response_json["tool"] == "add_numbers":
            params = response_json["parameters"]
            result = add_numbers(params["a"], params["b"])
            return f"Result: {result}"
    except json.JSONDecodeError:
        # If response is not JSON, it's a direct answer
        return response

# Example usage
if __name__ == "__main__":
    # Test cases
    queries = [
        "What is 5 + 3?",
        "What's the weather like today?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = process_query(query)
        print(f"Response: {result}")