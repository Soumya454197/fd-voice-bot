
import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

for attempt in range(1, 4):
    start = time.perf_counter()

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Reply in one short sentence: What is an FD?",
        )

        elapsed = time.perf_counter() - start
        print(f"Test {attempt}: {elapsed:.2f} seconds")
        print(response.text)
        print()

    except Exception as error:
        print(f"Test {attempt} failed: {error}")