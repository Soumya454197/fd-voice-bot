import asyncio
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types


# =============== CONFIGURATION ===============

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

KNOWLEDGE_FILE = Path("data/fd_knowledge.md")
PROMPT_FILE = Path("data/system_prompt.txt")

knowledge = (
    KNOWLEDGE_FILE.read_text(encoding="utf-8")
    if KNOWLEDGE_FILE.exists()
    else "No approved FD knowledge file was found."
)

base_prompt = (
    PROMPT_FILE.read_text(encoding="utf-8")
    if PROMPT_FILE.exists()
    else "You are an AI assistant for fictional XYZ Bank."
)

SYSTEM_INSTRUCTION = f"""
{base_prompt}

APPROVED FIXED DEPOSIT KNOWLEDGE:
{knowledge}

VOICE CONVERSATION RULES:
- You are the AI voice assistant for fictional XYZ Bank.
- Greet the caller once, then wait for their response.
- Maintain a natural, ongoing conversation.
- After each answer, continue listening for the next question.
- Do not repeat the greeting after every question.
- Give concise answers that are easy to understand by phone.
- Handle follow-up questions using the approved FD knowledge.
- Ask one question at a time when collecting callback details.
- Never ask for OTPs, PINs, passwords, card details, or full account numbers.
- Never invent rates, policies, or financial calculations.
- Offer a human representative for account-specific questions.
- Do not claim to open an FD or transfer money.
- If information is unavailable, say so honestly.
- Mention that rates and terms can change and should be confirmed with
  the bank.
- If the caller clearly says goodbye, politely close the conversation.
"""

GREETING_PROMPT = (
    "Greet the user once as the AI assistant for XYZ Bank. "
    "Ask how you can help with the long-term fixed deposit. "
    "Then wait for the user to speak. Do not end the conversation "
    "after the greeting."
)

client = genai.Client(api_key=API_KEY)
app = FastAPI()


# =============== BROWSER -> GEMINI ===============

async def browser_to_gemini(ws: WebSocket, session):
    """Receive mic audio (16 kHz PCM16 bytes) from the browser."""
    while True:
        message = await ws.receive()

        if message["type"] == "websocket.disconnect":
            return

        data = message.get("bytes")
        if data:
            await session.send_realtime_input(
                audio=types.Blob(
                    data=data,
                    mime_type="audio/pcm;rate=16000",
                )
            )


# =============== GEMINI -> BROWSER ===============

async def gemini_to_browser(ws: WebSocket, session, save_chat):
    """Send bot audio (24 kHz PCM16 bytes) and transcripts to the browser."""
    while True:
        # receive() can end after each turn; re-enter on the same session.
        async for response in session.receive():
            content = response.server_content
            if content is None:
                continue

            incoming = getattr(content, "input_transcription", None)
            if incoming and getattr(incoming, "text", None):
                save_chat("You", incoming.text)
                await ws.send_json(
                    {"type": "user_text", "text": incoming.text}
                )

            outgoing = getattr(content, "output_transcription", None)
            if outgoing and getattr(outgoing, "text", None):
                save_chat("XYZ Bank Assistant", outgoing.text)
                await ws.send_json(
                    {"type": "bot_text", "text": outgoing.text}
                )

            if getattr(content, "interrupted", False):
                await ws.send_json({"type": "interrupted"})

            turn = getattr(content, "model_turn", None)
            if turn:
                for part in turn.parts:
                    if part.inline_data and part.inline_data.data:
                        await ws.send_bytes(part.inline_data.data)

        await asyncio.sleep(0.05)


# =============== WEBSOCKET ENDPOINT ===============

@app.websocket("/ws")
async def voice_socket(ws: WebSocket):
    await ws.accept()

    log_path = LOG_DIR / (
        "web_call_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    )

    def save_chat(speaker, text):
        if not text or not text.strip():
            return
        line = (
            f"{datetime.now().strftime('%H:%M:%S')} | "
            f"{speaker}: {text.strip()}"
        )
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_INSTRUCTION,
        input_audio_transcription={},
        output_audio_transcription={},
    )

    try:
        async with client.aio.live.connect(
            model=MODEL, config=config
        ) as session:
            save_chat("System", "Web call started.")
            await ws.send_json({"type": "status", "text": "connected"})

            await session.send_realtime_input(text=GREETING_PROMPT)

            tasks = [
                asyncio.create_task(browser_to_gemini(ws, session)),
                asyncio.create_task(
                    gemini_to_browser(ws, session, save_chat)
                ),
            ]

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            for task in done:
                if not task.cancelled() and task.exception():
                    raise task.exception()

    except WebSocketDisconnect:
        pass
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        try:
            await ws.send_json({"type": "error", "text": str(error)})
        except Exception:
            pass
    finally:
        save_chat("System", "Web call ended.")
        print(f"Transcript saved to: {log_path.resolve()}", flush=True)


# =============== SERVE THE WEB PAGE ===============
# Keep this LAST so it doesn't override the /ws route.

app.mount("/", StaticFiles(directory="static", html=True), name="static")