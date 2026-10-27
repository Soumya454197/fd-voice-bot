from dotenv import load_dotenv
from google import genai

load_dotenv()                 # reads GEMINI_API_KEY from your .env file
client = genai.Client()       # the client finds the key by itself

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Say hello in one short sentence, as a friendly bank assistant.",
)
print(response.text)