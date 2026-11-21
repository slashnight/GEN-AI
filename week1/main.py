from dotenv import load_dotenv
from google import genai
from google.genai import errors
import os

# Step 1: Load API key from .env
load_dotenv()

# Step 2: Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY", "").strip()

if not api_key:
    raise ValueError("Gemini API key not found!")

# Step 3: Initialize Gemini client
client = genai.Client(api_key=api_key)

# Step 4: Make first LLM call
try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="What is Artificial Intelligence? Explain in simple words."
    )
except errors.ClientError as error:
    if error.code == 401:
        raise RuntimeError(
            "Gemini rejected GEMINI_API_KEY. Replace the value in week1/.env with a "
            "valid Google AI Studio API key; do not use a Google Cloud access token."
        ) from error
    raise

# Step 5: Display response
print("\nLLM Response:")
print(response.text)
