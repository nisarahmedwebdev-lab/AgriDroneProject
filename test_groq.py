"""Test Groq API connection (Free)."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ No Groq API key found in .env file")
    print("Please add: GROQ_API_KEY=gsk_...")
    print("Get one from: https://console.groq.com/keys")
    exit()

if not api_key.startswith("gsk_"):
    print("❌ Invalid API key format. Key should start with 'gsk_'")
    exit()

print("⏳ Testing Groq API (Free)...")

try:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": "Say hello in 5 words"}
        ],
        "max_tokens": 20,
        "temperature": 0.5
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Groq API works! (Free)")
        print(f"Response: {result['choices'][0]['message']['content']}")
        print("\n✅ You can now run: streamlit run app.py")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error: {e}")