
from dotenv import load_dotenv
from function_gemma import FunctionGemmaCaller
import torch

# Load environment variables from the root .env file
load_dotenv(override=True)

def main():
    # Define a sample tool (function definition)
    # Using the correct schema format with "type": "function" wrapper
    tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    }
    ]

    print("Initializing FunctionGemmaCaller...")
    # You can change model_id if needed, e.g., to a larger version if available
    agent = FunctionGemmaCaller(model_id="google/functiongemma-270m-it", device="cuda", dtype=torch.float16)
    
    query = "What is the weather in New York today?"
    print(f"\nUser Query: {query}")
    
    messages = [
        {"role": "user", "content": query}
    ]
    
    try:
        response = agent.run_inference(messages, tools)
        print("\nModel Response (Function Call):")
        print(response)
    except Exception as e:
        print(f"\nError during generation: {e}")

if __name__ == "__main__":
    main()
