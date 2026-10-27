
import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from your .env file."
    )

# Connect to Gemini
client = genai.Client(api_key=API_KEY)

# --------------------------------------------------
# 2. CONNECT TO CHROMADB
# --------------------------------------------------

DB_PATH = "chroma_db"
COLLECTION_NAME = "fd_knowledge"

db = chromadb.PersistentClient(path=DB_PATH)
collection = db.get_collection(name=COLLECTION_NAME)

# --------------------------------------------------
# 3. LOAD SYSTEM INSTRUCTIONS
# --------------------------------------------------

PROMPT_PATH = Path("data/system_prompt.txt")

if PROMPT_PATH.exists():
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
else:
    system_prompt = """
You are the AI assistant for fictional XYZ Bank.
Answer only questions about the approved Fixed Deposit scheme.
Use only the approved FD information provided to you.
Never invent rates, rules, or calculations.
If information is unavailable, offer a human representative.
Do not greet the customer again in every response.
Keep answers concise, usually 1-2 short sentences.
Never request OTPs, PINs, passwords, or card details.
"""

# --------------------------------------------------
# 4. COMMON QUESTIONS: INSTANT FAQ ANSWERS
# --------------------------------------------------

FAQ = {
    "what is the interest rate":
        "The standard interest rate is 4% per year. "
        "Senior citizens receive an additional 0.50% per year.",

    "what is the minimum amount":
        "The minimum deposit is ₹10,000.",

    "what is the minimum deposit":
        "The minimum deposit is ₹10,000.",

    "how long is the tenure":
        "The tenure ranges from 20 to 40 years.",

    "what is the tenure":
        "The tenure ranges from 20 to 40 years.",

    "what is the senior citizen benefit":
        "Senior citizens receive an additional 0.50% annual interest.",

    "can i take a loan against my fd":
        "You can obtain a loan of up to 90% of your FD deposit.",

    "what is the minimum tenure":
        "The minimum tenure is 20 years.",

    "what is the maximum tenure":
        "The maximum tenure is 40 years.",

    "how is interest compounded":
        "Interest is compounded quarterly and paid at maturity.",

    "when is interest paid":
        "Interest is compounded quarterly and paid at maturity.",

    "is this a real bank":
        "XYZ Bank is a fictional bank used for this project.",

    "hello":
        "Hello! I'm the AI assistant for XYZ Bank. "
        "I can answer questions about our Fixed Deposit scheme.",

    "hi":
        "Hello! I can help with questions about the XYZ Bank Fixed Deposit scheme.",

    "hey":
        "Hello! How can I help you with the Fixed Deposit scheme?",
}

# --------------------------------------------------
# 5. NORMALIZE QUESTIONS
# --------------------------------------------------

def normalize_question(question):
    normalized = question.lower().strip()
    normalized = " ".join(normalized.split())
    normalized = normalized.rstrip("?.! ")
    return normalized


# --------------------------------------------------
# 6. GET ANSWER
# --------------------------------------------------

def get_answer(question):
    start = time.perf_counter()
    normalized = normalize_question(question)

    # Instant exact-match FAQ answers
    if normalized in FAQ:
        print(
            f"[FAQ] Answered in "
            f"{time.perf_counter() - start:.3f}s"
        )
        return FAQ[normalized]

    # Handle common greetings
    if normalized in (
        "good morning",
        "good afternoon",
        "good evening",
    ):
        return (
            "Hello! I'm the AI assistant for XYZ Bank. "
            "I can help with questions about our Fixed Deposit scheme."
        )

    # Retrieve the two most relevant knowledge chunks
    print("[INFO] Searching FD knowledge...", flush=True)

    retrieval_start = time.perf_counter()

    try:
        results = collection.query(
            query_texts=[question],
            n_results=2,
        )

        documents = results.get("documents", [[]])[0]

    except Exception as error:
        print(f"[RETRIEVAL ERROR] {error}", flush=True)
        return (
            "I'm unable to access the approved FD information right now. "
            "Would you like a human bank representative to assist you?"
        )

    print(
        f"[TIMING] Retrieval: "
        f"{time.perf_counter() - retrieval_start:.2f}s",
        flush=True,
    )

    if not documents:
        return (
            "I couldn't find that information in our approved FD details. "
            "Would you like a human bank representative to assist you?"
        )

    context = "\n\n".join(documents)

    # --------------------------------------------------
    # 7. STREAM THE GEMINI RESPONSE
    # --------------------------------------------------

    print("[INFO] Generating answer...", flush=True)

    api_start = time.perf_counter()
    answer_parts = []

    try:
        stream = client.models.generate_content_stream(
            model="gemini-3.5-flash-lite",
            contents=f"""
Approved FD information:
{context}

Customer question:
{question}

Instructions:
- Answer the customer's specific question directly.
- Use only facts supported by the approved information.
- Do not greet or introduce yourself again.
- Do not add unrelated facts.
- Keep the answer to 1-2 short sentences.
- Do not add a follow-up question unless useful.
- If the answer is unsupported, say you are unsure
  and offer a human bank representative.
""",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=100,
            ),
        )

        # Print each text chunk as it arrives
        print("\nAssistant: ", end="", flush=True)

        first_chunk_time = None

        for chunk in stream:
            if chunk.text:
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                    print(
                        "\n[TIMING] Time to first text: "
                        f"{first_chunk_time - api_start:.2f}s",
                        flush=True,
                    )
                    print("Assistant: ", end="", flush=True)

                print(chunk.text, end="", flush=True)
                answer_parts.append(chunk.text)

        print("\n", flush=True)

        answer = "".join(answer_parts).strip()

        print(
            f"[TIMING] Total Gemini time: "
            f"{time.perf_counter() - api_start:.2f}s",
            flush=True,
        )

        print(
            f"[TIMING] Total processing time: "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

        if answer:
            return answer

        return (
            "I'm unable to answer that right now. "
            "Would you like a human bank representative to assist you?"
        )

    except Exception as error:
        print(f"\n[GEMINI ERROR] {error}", flush=True)

        return (
            "I'm unable to answer that right now. "
            "Would you like a human bank representative to assist you?"
        )


# --------------------------------------------------
# 8. CHAT LOOP
# --------------------------------------------------

if __name__ == "__main__":
    print("=" * 45)
    print("       XYZ BANK FD ASSISTANT")
    print("=" * 45)
    print("Ask questions about the Fixed Deposit scheme.")
    print("Type 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if not question:
                continue

            answer = get_answer(question)

            # Streaming answers have already been printed.
            # Print only FAQ or other non-streamed answers here.
            normalized = normalize_question(question)

            is_faq = (
                normalized in FAQ
                or normalized in (
                    "good morning",
                    "good afternoon",
                    "good evening",
                )
            )

            if is_faq:
                print(f"Assistant: {answer}")

            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as error:
            print(f"[ERROR] {error}", flush=True)
