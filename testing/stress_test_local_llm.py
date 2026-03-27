"""
stress_test_local_llm.py - Tests the effective context window size of LM Studio.
"""
import time
from openai import OpenAI
import re

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def generate_filler(target_tokens: int) -> str:
    """Generates roughly 'target_tokens' worth of filler text."""
    # 1 token is roughly 0.75 words. So 1 word is roughly 1.33 tokens.
    words = int(target_tokens * 0.75)
    filler_sentence = "The data scientist analyzed the variables carefully, ensuring constraints were met. "
    # 11 words per sentence
    num_sentences = words // 11
    return filler_sentence * num_sentences

def run_context_test(token_size: int) -> bool:
    print(f"\n--- Testing ~{token_size} tokens context window ---")
    
    magic_word = "GRAPEFRUIT"
    
    # We put the magic word at the very beginning of the prompt.
    prompt = f"System Note: The secret magic word is {magic_word}. Remember it.\n\n"
    prompt += generate_filler(token_size)
    prompt += "\n\nQuestion: Based on the very first sentence of this text, what is the secret magic word?"
    
    try:
        start = time.time()
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Low temperature for factual retrieval
            max_tokens=50
        )
        end = time.time()
        
        answer = response.choices[0].message.content.strip()
        print(f"Model Answer: {answer}")
        print(f"Time Taken: {round(end - start, 2)}s")
        
        # Check if the model successfully retrieved the word
        if magic_word.lower() in answer.lower():
            print("✅ PASS: Model successfully retrieved information from the start of the context.")
            return True
        else:
            print("❌ FAIL: Model forgot or truncated the beginning of the prompt.")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: API Call Failed. Context might be too large for LM Studio. Details: {e}")
        return False

def main():
    print("======================================================")
    print("  LM Studio Context Window & Memory Stress Tester")
    print("======================================================")
    print("This script will test how much text your local model can 'remember'")
    print("by hiding a magic word at the beginning of a massive document.")
    
    # Test incremental token sizes
    test_sizes = [500, 1000, 2000, 3000, 4000, 6000, 8000]
    
    for size in test_sizes:
        success = run_context_test(size)
        if not success:
            print(f"\n⚠️ WARNING: Your model's effective memory broke at ~{size} tokens.")
            print("This means you must keep all prompts (Data Profiles, JSONs) below this size.")
            break
        
        time.sleep(2) # Brief pause between calls

    print("\nStress Test Complete!")

if __name__ == "__main__":
    main()
