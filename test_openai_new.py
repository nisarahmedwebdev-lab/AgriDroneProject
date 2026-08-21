# test_openai_new.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def test_openai():
    print("=" * 60)
    print("Testing OpenAI API with New Key")
    print("=" * 60)
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("\nPlease add OPENAI_API_KEY=your_key_here to .env")
        return False
    
    print(f"✅ API Key found: {api_key[:15]}...{api_key[-5:]}")
    
    if not api_key.startswith('sk-'):
        print("❌ API Key format is incorrect")
        print("OpenAI keys should start with 'sk-'")
        return False
    
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, AgriDrone is working!'"}
            ],
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=50,
        )
        
        print("\n✅ OpenAI API is working!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        # Check if it's a quota error
        if "insufficient_quota" in str(e):
            print("\n⚠️ This account also has insufficient quota.")
            print("You need to add credits to your OpenAI account.")
            print("Or try using Hugging Face (FREE) instead.")
        elif "invalid_api_key" in str(e):
            print("\n⚠️ Invalid API key. Please check your key.")
        
        return False

if __name__ == "__main__":
    test_openai()