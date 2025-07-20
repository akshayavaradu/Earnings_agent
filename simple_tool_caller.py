import requests
import json
import pandas as pd

# Ollama chat API endpoint (assuming Ollama is running locally)
OLLAMA_CHAT_API_URL = "http://localhost:11434/api/chat"

# Define the tool to filter bank data from Excel
def filter_bank_metrics(banks: list, quarters: list, metrics: list) -> dict:
    """Filters metrics for specified banks and quarters from an Excel file."""
    try:
        # Read Excel file (adjust path as needed)
        df = pd.read_excel("bank_data.xlsx")
        
        # Validate inputs
        if not all(col in df.columns for col in ['Bank', 'Quarter'] + metrics):
            return {"error": "Invalid bank, quarter, or metric name"}
        
        # Filter data
        filtered_df = df[df['Bank'].isin(banks) & df['Quarter'].isin(quarters)][['Bank', 'Quarter'] + metrics]
        
        # Convert to dictionary
        result = filtered_df.to_dict(orient="records")
        return {"data": result} if result else {"error": "No data found for the specified banks, quarters, and metrics"}
    except FileNotFoundError:
        return {"error": "Excel file 'bank_data.xlsx' not found"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Available tools (metadata for the model)
TOOLS = [
    {
        "name": "filter_bank_metrics",
        "description": "Filters metrics for specified banks and quarters from an Excel file.",
        "parameters": {
            "banks": {"type": "list", "description": "List of bank names (e.g., ['Bank A', 'Bank B'])"},
            "quarters": {"type": "list", "description": "List of quarters (e.g., ['Q1 2023', 'Q2 2023'])"},
            "metrics": {"type": "list", "description": "List of metrics to retrieve (e.g., ['Revenue', 'Profit'])"}
        }
    }
]

# Function to call Ollama chat API
def call_ollama_chat(messages, model="llama3.1"):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    response = requests.post(OLLAMA_CHAT_API_URL, json=payload)
    if response.status_code == 200:
        return json.loads(response.text).get("message", {}).get("content", "")
    else:
        raise Exception(f"Ollama API error: {response.status_code}")

# Function to process user input and handle tool calling
def process_chat_query(user_input):
    # System prompt with tool information
    tool_info = json.dumps(TOOLS, indent=2)
    system_prompt = f"""
You are a helpful assistant with access to the following tools:
{tool_info}

For queries requiring a tool, respond with a JSON object:
- "tool": the name of the tool to call
- "parameters": the parameters for the tool

For conversational queries, respond with a plain string answer.

Respond concisely and accurately.
"""

    # Create message history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    # Call the Ollama chat API
    response = call_ollama_chat(messages)
    
    try:
        # Try to parse the response as JSON (for tool calls)
        response_json = json.loads(response)
        if "tool" in response_json and response_json["tool"] == "filter_bank_metrics":
            params = response_json["parameters"]
            result = filter_bank_metrics(params["banks"], params["quarters"], params["metrics"])
            if "error" in result:
                return f"Error: {result['error']}"
            return json.dumps(result["data"], indent=2)
    except json.JSONDecodeError:
        # If response is not JSON, it's a conversational answer
        return response

# Example usage with a simple chat loop
if __name__ == "__main__":
    print("Start chatting (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        print(f"Assistant: {process_chat_query(user_input)}")