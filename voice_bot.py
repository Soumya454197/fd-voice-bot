
import asyncio
import os
import queue
import threading
import audioop
from datetime import datetime
from pathlib import Path

import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types


# =============== CONFIGURATION ===============

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

# Leave as None to use Windows' default microphone.
# If necessary, set this to a microphone index shown at startup.
MIC_DEVICE_INDEX = None

INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 1024

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

client = genai.Client(api_key=API_KEY)

# Thread-safe queues: microphone and speaker use background threads.
mic_queue = queue.Queue(maxsize=30)
speaker_queue = queue.Queue(maxsize=100)

stop_event = threading.Event()

log_path = LOG_DIR / (
    "voice_chat_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
)


# =============== TRANSCRIPT ===============

def save_chat(speaker, text):
    if not text or not text.strip():
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"{timestamp} | {speaker}: {text.strip()}"

    print(line, flush=True)

    with log_path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


# =============== AUDIO DEVICE LIST ===============

def print_audio_devices():
    print("\nAvailable audio devices:")

    for index in range(audio.get_device_count()):
        device = audio.get_device_info_by_index(index)

        if (
            device["maxInputChannels"] > 0
            or device["maxOutputChannels"] > 0
        ):
            print(
                f"{index}: {device['name']} | "
                f"inputs={device['maxInputChannels']} | "
                f"outputs={device['maxOutputChannels']}"
            )


# =============== MICROPHONE THREAD ===============

def capture_microphone():
    stream = None

    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=INPUT_RATE,
            input=True,
            input_device_index=MIC_DEVICE_INDEX,
            frames_per_buffer=CHUNK_SIZE,
        )

        print("Microphone capture started.", flush=True)
        counter = 0

        while not stop_event.is_set():
            data = stream.read(
                CHUNK_SIZE,
                exception_on_overflow=False,
            )

            # Avoid building up a large backlog of old audio.
            try:
                mic_queue.put_nowait(data)
            except queue.Full:
                try:
                    mic_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    mic_queue.put_nowait(data)
                except queue.Full:
                    pass

            counter += 1
            if counter % 30 == 0:
                print(
                    f"Microphone level: {audioop.rms(data, 2)}",
                    flush=True,
                )

    except Exception as error:
        if not stop_event.is_set():
            print(
                f"Microphone error: {type(error).__name__}: {error}",
                flush=True,
            )
            raise

    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()


# =============== SEND MICROPHONE AUDIO ===============

async def send_audio(session):
    print("Audio sender started.", flush=True)

    while not stop_event.is_set():
        try:
            # queue.Queue is thread-safe; don't access asyncio.Queue
            # directly from a background thread.
            data = await asyncio.to_thread(mic_queue.get, True, 0.25)
        except queue.Empty:
            continue

        await session.send_realtime_input(
            audio=types.Blob(
                data=data,
                mime_type="audio/pcm;rate=16000",
            )
        )


# =============== SPEAKER THREAD ===============

def play_audio():
    stream = None

    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=OUTPUT_RATE,
            output=True,
        )

        print("Speaker playback started.", flush=True)

        while not stop_event.is_set():
            try:
                data = speaker_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            stream.write(data)

    except Exception as error:
        if not stop_event.is_set():
            print(
                f"Speaker error: {type(error).__name__}: {error}",
                flush=True,
            )
            raise

    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()


# =============== RECEIVE GEMINI RESPONSES ===============

async def receive_audio(session):
    print("Waiting for Gemini responses.", flush=True)

    while not stop_event.is_set():
        try:
            # Gemini's receive iterator can end after one response turn.
            # Re-enter it on the SAME session to receive the next turn.
            async for response in session.receive():
                content = response.server_content

                if content is None:
                    continue

                # User speech transcription
                incoming = getattr(
                    content, "input_transcription", None
                )
                if incoming:
                    text = getattr(incoming, "text", None)
                    if text:
                        save_chat("You", text)

                # Assistant speech transcription
                outgoing = getattr(
                    content, "output_transcription", None
                )
                if outgoing:
                    text = getattr(outgoing, "text", None)
                    if text:
                        save_chat("XYZ Bank Assistant", text)

                # Stop queued speech if the user interrupts.
                if getattr(content, "interrupted", False):
                    while True:
                        try:
                            speaker_queue.get_nowait()
                        except queue.Empty:
                            break

                # Process every audio part in this event.
                turn = getattr(content, "model_turn", None)

                if turn:
                    for part in turn.parts:
                        if part.inline_data and part.inline_data.data:
                            try:
                                speaker_queue.put_nowait(
                                    part.inline_data.data
                                )
                            except queue.Full:
                                save_chat(
                                    "System",
                                    "Speaker queue full; audio chunk skipped.",
                                )

            if not stop_event.is_set():
                print(
                    "Response turn finished; continuing to listen...",
                    flush=True,
                )
                await asyncio.sleep(0.05)

        except asyncio.CancelledError:
            raise

        except Exception as error:
            save_chat(
                "System",
                f"Gemini receive error: {type(error).__name__}: {error}",
            )
            raise


# =============== MAIN VOICE SESSION ===============

async def run_voice_assistant():
    tasks = []

    try:
        print_audio_devices()

        print(f"\nTranscript will be saved to:\n{log_path.resolve()}")
        print(f"\nConnecting to Gemini Live model: {MODEL}", flush=True)

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_INSTRUCTION,
            input_audio_transcription={},
            output_audio_transcription={},
        )

        async with client.aio.live.connect(
            model=MODEL,
            config=config,
        ) as session:

            print("\nConnected successfully!", flush=True)
            save_chat("System", "Voice assistant session started.")

            # Start the microphone and speaker in background threads.
            mic_task = asyncio.create_task(
                asyncio.to_thread(capture_microphone)
            )
            speaker_task = asyncio.create_task(
                asyncio.to_thread(play_audio)
            )

            sender_task = asyncio.create_task(send_audio(session))
            receiver_task = asyncio.create_task(receive_audio(session))

            tasks = [
                mic_task,
                speaker_task,
                sender_task,
                receiver_task,
            ]

            # Send one greeting request. The session stays open afterward.
            await session.send_realtime_input(
                text=(
                    "Greet the user once as the AI assistant for XYZ Bank. "
                    "Ask how you can help with the long-term fixed deposit. "
                    "Then wait for the user to speak. Do not end the "
                    "conversation after the greeting."
                )
            )

            print(
                "\nSpeak naturally. Ask multiple questions without "
                "restarting the program. Press Ctrl+C to stop.\n",
                flush=True,
            )

            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # A background task stopping unexpectedly is an error.
            for task in done:
                if not task.cancelled():
                    error = task.exception()
                    if error:
                        raise error

            if not stop_event.is_set():
                raise RuntimeError(
                    "A voice task stopped unexpectedly. "
                    "See the error printed above."
                )

    except asyncio.CancelledError:
        pass

    except Exception as error:
        print(
            f"\nERROR: {type(error).__name__}: {error}",
            flush=True,
        )

    finally:
        stop_event.set()

        for task in tasks:
            if not task.done():
                task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        try:
            await asyncio.to_thread(audio.terminate)
        except Exception:
            pass

        save_chat("System", "Voice assistant session ended.")
        print(f"\nTranscript saved to:\n{log_path.resolve()}", flush=True)


if __name__ == "__main__":
    audio = pyaudio.PyAudio()

    try:
        asyncio.run(run_voice_assistant())
    except KeyboardInterrupt:
        stop_event.set()
        print("\nVoice assistant stopped.", flush=True)
    finally:
        try:
            audio.terminate()
        except Exception:
            pass
