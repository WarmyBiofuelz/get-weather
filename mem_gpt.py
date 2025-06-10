# This will be a console application that uses the OpenAI API to create a memory-augmented GPT model.
# The application will run in a loop and allow the user to interact with the model.
# if user wants to exit, they can type 'exit' or 'quit'.
# The model will have access to two tools/function calls:
# 1. `write_to_memory`: This will write the user's input to memory.
# 2. `read_from_memory`: This will read the contents of memory and return it.
# The memory will be stored in a file called `memory.txt`.
# this will be implemented using the OpenAI API's function calling feature.

import os
import json
from openai import OpenAI
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

token = os.getenv("SECRET2")  # Using SECRET2 for GitHub API
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

client = OpenAI(
    base_url=endpoint,
    api_key=token
)

# Memory file path
MEMORY_FILE = "memory.txt"

def write_to_memory(content: str) -> str:
    """Write content to memory file with timestamp."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {content}\n")
        return f"Successfully wrote to memory: {content}"
    except Exception as e:
        return f"Error writing to memory: {str(e)}"

def read_from_memory() -> str:
    """Read all contents from memory file."""
    try:
        if not os.path.exists(MEMORY_FILE):
            return "Memory is empty. No previous entries found."
        
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if not content:
            return "Memory is empty."
        
        return f"Memory contents:\n{content}"
    except Exception as e:
        return f"Error reading from memory: {str(e)}"

# Function definitions for OpenAI function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "write_to_memory",
            "description": "Write important information to memory for future reference",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to write to memory"
                    }
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_from_memory",
            "description": "Read all previously stored information from memory",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

def handle_function_call(function_name: str, arguments: dict) -> str:
    """Handle function calls from the AI model."""
    if function_name == "write_to_memory":
        return write_to_memory(arguments.get("content", ""))
    elif function_name == "read_from_memory":
        return read_from_memory()
    else:
        return f"Unknown function: {function_name}"

def main():
    """Main application loop."""
    print("Memory-Augmented GPT Console Application")
    print("========================================")
    print("I'm an AI assistant with memory capabilities.")
    print("I can remember important information from our conversation.")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # Initialize conversation history
    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI assistant with memory capabilities. 
            
            You can remember important information from the conversation and recall it later.
            You can write to memory using the 'write_to_memory' function 
            and read from memory using the 'read_from_memory' function."""
        }
    ]
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("AI: Goodbye! It was nice talking with you.")
                break
            
            # Add user message to conversation
            messages.append({"role": "user", "content": user_input})
            
            # Make API call with function calling
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7
            )
            
            response_message = response.choices[0].message
            
            # Check if the model wants to call a function
            if response_message.tool_calls:
                # Add the assistant's response to messages
                messages.append(response_message)
                
                # Process each tool call
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the function
                    function_result = handle_function_call(function_name, function_args)
                    
                    # Add function result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": function_result
                    })
                
                # Get final response after function execution
                final_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7
                )
                
                ai_response = final_response.choices[0].message.content
            else:
                ai_response = response_message.content
            
            # Add AI response to conversation and display
            messages.append({"role": "assistant", "content": ai_response})
            print(f"AI: {ai_response}")
            
        except KeyboardInterrupt:
            print("\n\nAI: Goodbye! Conversation interrupted.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again.")
    
main()