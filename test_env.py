#!/usr/bin/env python3
"""
Quick test to verify .env is loaded
"""
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

print("=" * 50)
print("🔍 CHECKING .ENV CONFIGURATION")
print("=" * 50)

# Check Groq API Key
groq_key = os.getenv('GROQ_API_KEY')
if groq_key:
    if groq_key.startswith('gsk_'):
        masked = f"{groq_key[:8]}...{groq_key[-4:]}"
        print(f"✅ GROQ_API_KEY: {masked} (Valid format)")
    else:
        print(f"⚠️  GROQ_API_KEY: Invalid format (should start with 'gsk_')")
else:
    print("❌ GROQ_API_KEY: Not found!")

# Check OpenAI API Key
openai_key = os.getenv('OPENAI_API_KEY')
if openai_key:
    if openai_key.startswith('sk-'):
        masked = f"{openai_key[:8]}...{openai_key[-4:]}"
        print(f"✅ OPENAI_API_KEY: {masked} (Valid format)")
    else:
        print(f"⚠️  OPENAI_API_KEY: Invalid format (should start with 'sk-')")
else:
    print("⚠️  OPENAI_API_KEY: Not found (OK if using Groq)")

# Check LLM Provider
llm_provider = os.getenv('LLM_PROVIDER', 'groq')
llm_model = os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')
print(f"📡 LLM_PROVIDER: {llm_provider}")
print(f"🤖 LLM_MODEL: {llm_model}")

# Check MongoDB
mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
print(f"🗄️  MONGODB_URI: {mongo_uri}")

print("=" * 50)

# Final verdict
if groq_key and groq_key.startswith('gsk_'):
    print("✅ Configuration is VALID!")
    print("🚀 You can now run: streamlit run frontend/login.py")
elif openai_key and openai_key.startswith('sk-'):
    print("✅ Configuration is VALID (using OpenAI)!")
    print("🚀 You can now run: streamlit run frontend/login.py")
else:
    print("❌ Configuration INVALID!")
    print("💡 Please check your .env file")
    print("   Example:")
    print("   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

print("=" * 50)