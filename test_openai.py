# test_openai.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def test_openai():
    print("=" * 60)
    print("Testing OpenAI API")
    print("=" * 60)
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("\nPlease add OPENAI_API_KEY=your_key_here to .env")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-5:]}")
    
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
        return False

if __name__ == "__main__":
    test_openai()