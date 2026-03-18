"""
test_api.py - Gemini API key and model tester with rate-limit awareness.
Run with: .\venv\Scripts\python test_api.py
"""
import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

KEYS = {
    "GEMINI_API_KEY20": os.getenv("GEMINI_API_KEY20"),
    "GEMINI_KEY1": os.getenv("GEMINI_KEY1"),
    "GEMINI_KEY2": os.getenv("GEMINI_KEY2"),
    "GEMINI_KEY3": os.getenv("GEMINI_KEY3"),
}

PROMPT = "Say hello in exactly 5 words."
DELAY_BETWEEN_CALLS = 5  # seconds between calls to avoid RPM limits

print("=" * 60)
print("AStats - Gemini API Key & Model Tester")
print("=" * 60)

# Step 1: List available models
first_key = next((v for v in KEYS.values() if v), None)
if first_key:
    print("\n[Step 1] Available flash/pro models:")
    try:
        client = genai.Client(api_key=first_key)
        models = list(client.models.list())
        flash_models = [
            m.name for m in models
            if "flash" in m.name.lower() or "pro" in m.name.lower()
        ]
        for m in sorted(flash_models):
            print(f"  {m}")
        test_models = [m.name.replace("models/", "") for m in models
                       if "flash" in m.name.lower() and "exp" not in m.name.lower()]
        test_models = sorted(set(test_models))[:4]
    except Exception as e:
        print(f"  Could not list models: {e}")
        test_models = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]
else:
    test_models = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]

print(f"\n[Step 2] Testing models: {test_models}")
print(f"         Delay between calls: {DELAY_BETWEEN_CALLS}s")
print("-" * 60)

# Step 2: Test each key
working = []

for key_name, key_value in KEYS.items():
    if not key_value:
        print(f"\n[{key_name}] SKIP - not set in .env")
        continue

    print(f"\n[{key_name}] Testing...")
    client = genai.Client(api_key=key_value)
    key_worked = False

    for model in test_models:
        print(f"  Trying {model} ...", end=" ", flush=True)
        time.sleep(DELAY_BETWEEN_CALLS)
        try:
            response = client.models.generate_content(model=model, contents=PROMPT)
            text = response.text.strip()
            print(f'OK  -> "{text}"')
            working.append((key_name, model))
            key_worked = True
            break
        except Exception as e:
            err_short = str(e)[:100]
            if "429" in str(e):
                print(f"RATE LIMITED - {err_short}")
            elif "404" in str(e):
                print(f"NOT FOUND    - {err_short}")
            else:
                print(f"ERROR        - {err_short}")

    if not key_worked:
        print(f"  [{key_name}] No working model found.")

# Summary
print("\n" + "=" * 60)
if working:
    print("Working combinations:")
    for k, m in working:
        print(f"  {k}  +  {m}")
    best_key, best_model = working[0]
    print(f'\nRecommended model: "{best_model}"')
else:
    print("No working key/model found.")
    print("\nPossible reasons:")
    print("  1. Free-tier daily quota exhausted - wait until midnight PST")
    print("  2. Free tier not available in your region")
    print("  3. All keys are in the same Google project (shared quota)")
    print("\nCheck your usage: https://ai.dev/rate-limit")
print("=" * 60)
