"""Test Groq API connection with current models."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ No Groq API key found. Get one from https://console.groq.com/keys")
    exit()

if not api_key.startswith("gsk_"):
    print("❌ Invalid key format. Key should start with 'gsk_'")
    exit()

# List of currently supported, free models to test
models_to_test = [
    "llama-3.3-70b-versatile",  # Most capable
    "llama-3.1-8b-instant",     # Fastest
    "gemma2-9b-it"              # Good alternative
]

print("⏳ Testing Groq API with different models...")

for model_name in models_to_test:
    print(f"\n🔄 Testing model: {model_name}")
    try:
        response = requests.post(
            url="https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say hello in 5 words"}],
                "max_tokens": 20
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS with {model_name}!")
            print(f"Response: {result['choices'][0]['message']['content']}")
            print(f"\n✅ Update MODEL_NAME in llm_handler.py to: {model_name}")
            break
        else:
            error = response.json().get('error', {})
            print(f"❌ Failed: {error.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")