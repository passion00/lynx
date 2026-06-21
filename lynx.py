import atexit
import subprocess
import time
from pathlib import Path
from lynx_core.active_memory import ActiveMemory
from lynx_core.database import LynxDatabase
import sys
import requests


# ===== Lynx configuration =====

LLAMA_CPP_DIR = Path.home() / "lynx" / "llama.cpp"
LLAMA_SERVER_BIN = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"

HOST = "127.0.0.1"
PORT = 8081

SERVER_URL = f"http://{HOST}:{PORT}"
CHAT_URL = f"{SERVER_URL}/v1/chat/completions"

MODEL_NAME = "llmware/qwen3-4b-instruct-gguf:Q4_K_M"

SERVER_COMMAND = [
    str(LLAMA_SERVER_BIN),
    "-hf", MODEL_NAME,
    "--host", HOST,
    "--port", str(PORT),
    "-c", "4096",
    "-ngl", "0",
]
SYSTEM_PROMPT = (
    "You are Lynx, a local AI assistant. "
    "Answer clearly, directly, and helpfully."
)

server_process: subprocess.Popen | None = None


def is_server_running() -> bool:
    """Check whether llama-server is already responding."""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=2)
        return response.status_code < 500
    except requests.RequestException:
        return False


def start_llama_server() -> None:
    """Start llama-server if it is not already running."""
    global server_process

    if is_server_running():
        print("Lynx model server is already running.")
        return

    if not LLAMA_SERVER_BIN.exists():
        raise FileNotFoundError(
            f"Could not find llama-server at:\n{LLAMA_SERVER_BIN}\n\n"
            "Make sure llama.cpp finished building successfully."
        )

    print("Starting Lynx model server...")
    print("This may take a long time on first run if the model is being downloaded.")

    log_file = open(Path.home() / "lynx" / "llama_server.log", "a")

    server_process = subprocess.Popen(
        SERVER_COMMAND,
        cwd=str(LLAMA_CPP_DIR),
        stdout=log_file,
        stderr=log_file,
        text=True,
    )

    seconds_waited = 0

    # Wait forever until the server becomes available,
    # unless llama-server crashes.
    while True:
        if is_server_running():
            print("Lynx model server is ready.")
            return

        if server_process.poll() is not None:
            raise RuntimeError(
                "llama-server stopped unexpectedly. Check this log file:\n"
                f"{Path.home() / 'lynx' / 'llama_server.log'}"
            )

        time.sleep(1)
        seconds_waited += 1

        if seconds_waited % 30 == 0:
            print("Still waiting for Lynx model server...")

def stop_llama_server() -> None:
    """Stop the llama-server process if this Python program started it."""
    global server_process

    if server_process is None:
        return

    if server_process.poll() is None:
        print("\nStopping Lynx model server...")
        server_process.terminate()

        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()


def ask_model(messages: list[dict[str, str]]) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.4,
    }

    response = requests.post(CHAT_URL, json=payload, timeout=None)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]

def read_user_input(prompt: str = "\nYou: ") -> str:
    """
    Read user input safely.

    This prevents Lynx from crashing if the terminal sends
    malformed UTF-8 bytes.
    """
    print(prompt, end="", flush=True)

    raw_input = sys.stdin.buffer.readline()

    if raw_input == b"":
        return "kill"

    return raw_input.decode("utf-8", errors="replace").strip()

def main() -> None:
    atexit.register(stop_llama_server)

    db = None
    conversation_id = None

    try:
        start_llama_server()

        memory = ActiveMemory(
            system_prompt=SYSTEM_PROMPT,
            max_messages=30,
        )

        db = LynxDatabase()
        conversation_id = db.start_conversation(title="Lynx chat")

        print("\nLynx controller started.")
        print('Type "kill" to exit.')
        print('Type "clear" to clear active memory.')
        print('Type "status" to show active memory status.')

        while True:
            user_input = read_user_input()

            if user_input.lower() == "kill":
                print("Lynx controller stopped.")
                break

            if user_input.lower() == "clear":
                memory.clear()
                print("Active memory cleared.")
                continue

            if user_input.lower() == "status":
                print(f"Active memory contains {memory.message_count()} messages.")
                print(f"Current conversation ID: {conversation_id}")
                print(f"Database path: {db.db_path}")
                continue

            if not user_input:
                continue

            try:
                memory.add_user_message(user_input)
                db.save_message(conversation_id, "user", user_input)

                answer = ask_model(memory.get_messages())

                memory.add_assistant_message(answer)
                db.save_message(conversation_id, "assistant", answer)

                print(f"\nLynx: {answer}")

            except requests.exceptions.ConnectionError:
                print("\nError: Could not connect to the Lynx model server.")
            except Exception as error:
                print(f"\nError: {error}")

    finally:
        if db is not None and conversation_id is not None:
            db.end_conversation(conversation_id)
            db.close()

if __name__ == "__main__":
    main()
