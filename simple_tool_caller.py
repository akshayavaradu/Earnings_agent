import pandas as pd
import json
import openai
from typing import Dict, Any, Callable

# Configure OpenAI API (set your API key)
openai.api_key = "your-openai-api-key"  # Replace with your actual API key

def filter_bank_data(query: str) -> Dict[str, Any]:
    """
    A tool that takes a JSON string containing bank names, quarters, and metrics,
    filters an Excel file, and returns the filtered data as JSON.
    
    Args:
        query (str): JSON string with format {'banks': [], 'quarters': [], 'metrics': []}
        
    Example Input:
        '{"banks": ["Bank A", "Bank B"], "quarters": ["1Q2025"], "metrics": ["Net Income", "Net Revenue"]}'
        
    Returns:
        Dict containing the filtered data or an error message.
    """
    try:
        # Parse the JSON input
        params = json.loads(query)
        
        # Validate input structure
        if not isinstance(params, dict):
            return {"error": "Input must be a JSON object"}
        
        banks = params.get("banks", [])
        quarters = params.get("quarters", [])
        metrics = params.get("metrics", [])
        
        # Validate required fields
        if not isinstance(banks, list) or not banks:
            return {"error": "Invalid or missing 'banks' field: must be a non-empty list"}
        if not isinstance(quarters, list) or not quarters:
            return {"error": "Invalid or missing 'quarters' field: must be a non-empty list"}
        if not isinstance(metrics, list) or not metrics:
            return {"error": "Invalid or missing 'metrics' field: must be a non-empty list"}
        
        # Validate field types
        if not all(isinstance(b, str) for b in banks):
            return {"error": "'banks' must contain only strings"}
        if not all(isinstance(q, str) for q in quarters):
            return {"error": "'quarters' must contain only strings"}
        if not all(isinstance(m, str) for m in metrics):
            return {"error": "'metrics' must contain only strings"}
        
        # Load and filter the Excel file
        try:
            # Replace 'data.xlsx' with the actual path to your Excel file
            df = pd.read_excel("data.xlsx")
            
            # Ensure the required columns exist
            required_columns = ['Bank', 'Quarter'] + metrics
            if not all(col in df.columns for col in required_columns):
                return {"error": "Excel file missing required columns"}
            
            # Filter the DataFrame
            filtered_df = df[
                (df['Bank'].isin(banks)) & 
                (df['Quarter'].isin(quarters))
            ][required_columns]
            
            # Convert filtered data to JSON
            result = filtered_df.to_dict(orient="records")
            return {"data": result}
        
        except FileNotFoundError:
            return {"error": "Excel file not found"}
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}
    
    except json.JSONDecodeError:
        return {"error": "Invalid JSON input"}
    except Exception as e:
        return {"error": f"Invalid input format: {str(e)}"}

def add_numbers(query: str) -> Dict[str, Any]:
    """
    A tool that takes a JSON string containing two numbers and returns their sum.
    
    Args:
        query (str): JSON string with format {'num1': float, 'num2': float}
        
    Example Input:
        '{"num1": 5, "num2": 10}'
        
    Returns:
        Dict containing the sum or an error message.
    """
    try:
        # Parse the JSON input
        params = json.loads(query)
        
        # Validate input structure
        if not isinstance(params, dict):
            return {"error": "Input must be a JSON object"}
        
        num1 = params.get("num1")
        num2 = params.get("num2")
        
        # Validate required fields
        if num1 is None or num2 is None:
            return {"error": "Missing required fields: 'num1' and 'num2'"}
        
        # Validate field types
        if not isinstance(num1, (int, float)) or not isinstance(num2, (int, float)):
            return {"error": "'num1' and 'num2' must be numbers"}
        
        # Calculate the sum
        result = num1 + num2
        return {"sum": result}
    
    except json.JSONDecodeError:
        return {"error": "Invalid JSON input"}
    except Exception as e:
        return {"error": f"Invalid input format: {str(e)}"}

# Dictionary of tools
tools = {
    "filter_bank_data": filter_bank_data,
    "add_numbers": add_numbers
}

def process_user_question(question: str) -> Dict[str, Any]:
    """
    Process a user question, determine which tool to call, extract parameters using a language model,
    and invoke the appropriate tool.
    
    Args:
        question (str): Natural language question, e.g.,
            "What are the net income and net revenue for Bank A and Bank B in 1Q2025?"
            or "Add 5 and 10"
    
    Returns:
        Dict containing the tool's output or an error message.
    """
    # Define the prompt to determine the tool and extract parameters
    prompt = """
    Analyze the user question and determine which tool to call from the following options:
    1. filter_bank_data: For questions about bank data, requiring bank names, quarters, and metrics.
    2. add_numbers: For questions about adding two numbers.
    
    Return a JSON object with the following structure:
    {
        "tool": "filter_bank_data" or "add_numbers",
        "parameters": string (JSON string with parameters for the selected tool)
    }
    
    For filter_bank_data, the parameters should be a JSON string like:
    '{"banks": [], "quarters": [], "metrics": []}'
    
    For add_numbers, the parameters should be a JSON string like:
    '{"num1": float, "num2": float}'
    
    Question: {question}
    
    If the question is unclear or doesn't match either tool, return:
    {"error": "Unable to determine appropriate tool or parameters"}
    """
    
    try:
        # Call the OpenAI API to parse the question
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {question}"}
            ],
            temperature=0
        )
        
        # Extract the response content
        result = json.loads(response.choices[0].message["content"])
        
        # Validate the response
        if "error" in result:
            return result
        
        tool_name = result.get("tool")
        parameters = result.get("parameters")
        
        if not tool_name or not parameters:
            return {"error": "Invalid response from language model"}
        
        # Invoke the appropriate tool from the dictionary
        if tool_name in tools:
            tool_func = tools[tool_name]
            return tool_func(parameters)
        else:
            return {"error": "Unknown tool specified"}
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Example usage
if __name__ == "__main__":
    # Example questions
    questions = [
        "What are the net income and net revenue for Bank A and Bank B in 1Q2025?",
        "Add 5 and 10"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = process_user_question(question)
        print(json.dumps(result, indent=2))