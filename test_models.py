import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List all models and print only those that support content generation
print("--- AVAILABLE MODELS FOR YOUR KEY ---")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)