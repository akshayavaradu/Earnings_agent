import json
import pandas as pd

def get_question_from_file():
    """Reads the user's question from a temporary file."""
    with open("temp_question.txt", "r") as f:
        return f.read().strip()

def extract_entities(question):
    """
    Extracts entities (banks, metrics, quarters) from the user's question.
    
    NOTE: This is a placeholder function. In a real implementation, this would
    use a call to an LLM to extract the entities.
    """
    # Placeholder implementation - replace with actual LLM call
    print("---")
    print(f"DEBUG: Bypassing LLM call and using hardcoded entities for now. Question: '{question}'")
    print("---")
    
    # Example of what the LLM should return:
    extracted_data = {
        "banks": ["Citigroup Inc", "JPMorgan Chase & Co"],
        "metrics": ["Net Income", "Total Revenue"],
        "quarters": ["2024 Q1", "2024 Q2"]
    }
    
    return extracted_data

def filter_data_from_excel(entities):
    """
    Reads the Excel file, filters the data based on the provided entities,
    and returns the filtered data as a JSON string.
    """
    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel("50_metrics.xlsx")
        
        # Filter the DataFrame based on the entities
        filtered_df = df[
            df["Bank Name"].isin(entities["banks"]) &
            df["Metric"].isin(entities["metrics"]) &
            df["Quarter"].isin(entities["quarters"])
        ]
        
        # Convert the filtered DataFrame to a JSON string
        result_json = filtered_df.to_json(orient="records")
        
        return result_json
        
    except FileNotFoundError:
        return json.dumps({"error": "The file '50_metrics.xlsx' was not found."})
    except Exception as e:
        return json.dumps({"error": str(e)})

def main():
    """Main function to run the script."""
    
    # 1. Get the user's question from the temporary file
    user_question = get_question_from_file()
    
    # 2. Extract entities from the question (using a placeholder for LLM)
    extracted_entities = extract_entities(user_question)
    
    # Print the extracted entities
    print("\nExtracted Entities:")
    print(json.dumps(extracted_entities, indent=4))
    
    # 3. Call the tool to filter the data
    filtered_json_data = filter_data_from_excel(extracted_entities)
    
    # 4. Print the final filtered data
    print("\nFiltered Data:")
    print(filtered_json_data)

if __name__ == "__main__":
    main()
