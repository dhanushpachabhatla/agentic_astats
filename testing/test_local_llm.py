"""
test_local_llm.py - Temporary script to test connection to LM Studio locally.
"""
import time
from openai import OpenAI

# Connect to LM Studio's local server
# Default port for LM Studio is 1234
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def test_connection():
    print("Testing connection to local LM Studio server...")
    try:
        start_time = time.time()
        
        # We ask a simple question to test the connection and the model's generation capabilities.
        response = client.chat.completions.create(
            model="Mistral-7B-Instruct-v0.1", # LM Studio usually ignores this and uses whatever model is loaded
            messages=[
                {"role": "system", "content": "You are a helpful data science assistant. Always respond concisely."},
                {"role": "user", "content": "What are the core assumptions of an Ordinary Least Squares (OLS) regression?"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        end_time = time.time()
        
        print("\n✅ Connection Successful!\n")
        print("--- Model Response ---")
        print(response.choices[0].message.content.strip())
        print("----------------------")
        print(f"\nResponse time: {round(end_time - start_time, 2)} seconds")
        print("Your local environment is ready for hybrid Agentic execution!")
        
    except Exception as e:
        print("\n❌ Connection Failed!")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Is LM Studio open?")
        print("2. Did you load the model (mistral-7b-instruct-v0.1.Q4_K_M.gguf)?")
        print("3. Did you start the Local Server? (Go to the <-> icon on the left tab and click 'Start Server')")

if __name__ == "__main__":
    test_connection()
