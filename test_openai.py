"""Test OpenAI API connection."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ No OpenAI API key found in .env file")
    print("Please add: OPENAI_API_KEY=sk-...")
    exit()

if not api_key.startswith("sk-"):
    print("❌ Invalid API key format. Key should start with 'sk-'")
    exit()

print("⏳ Testing OpenAI API...")

try:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Say hello in 5 words"}
        ],
        "max_tokens": 20,
        "temperature": 0.5
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ OpenAI API works!")
        print(f"Response: {result['choices'][0]['message']['content']}")
        print("\n✅ You can now run: streamlit run app.py")
    else:
        error = response.json().get('error', {})
        print(f"❌ Error: {response.status_code}")
        print(f"Message: {error.get('message', 'Unknown error')}")
        
        if response.status_code == 401:
            print("\n💡 Tip: Your API key is invalid. Check it and try again.")
        elif response.status_code == 429:
            print("\n💡 Tip: You've exceeded your quota. Check your billing.")
        elif response.status_code == 402:
            print("\n💡 Tip: Insufficient credits. Add payment method.")

except Exception as e:
    print(f"❌ Error: {e}")