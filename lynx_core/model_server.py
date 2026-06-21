"""
model_server.py

Controls the local llama.cpp server used by Lynx.

Responsibilities:
- define model/server configuration
- check whether llama-server is running
- start llama-server when Lynx starts
- stop llama-server when Lynx exits, if Lynx started it
"""

import subprocess
import time
from pathlib import Path
from typing import TextIO

import requests


# ===== llama.cpp / model configuration =====

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


server_process: subprocess.Popen | None = None
server_log_file: TextIO | None = None


def is_server_running() -> bool:
    """
    Check whether llama-server is already responding.
    """
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=2)
        return response.status_code < 500
    except requests.RequestException:
        return False


def start_llama_server() -> None:
    """
    Start llama-server if it is not already running.

    This function waits forever until the server is ready,
    unless the llama-server process crashes.
    """
    global server_process
    global server_log_file

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

    log_path = Path.home() / "lynx" / "llama_server.log"
    server_log_file = open(log_path, "a", encoding="utf-8")

    server_process = subprocess.Popen(
        SERVER_COMMAND,
        cwd=str(LLAMA_CPP_DIR),
        stdout=server_log_file,
        stderr=server_log_file,
        text=True,
    )

    seconds_waited = 0

    while True:
        if is_server_running():
            print("Lynx model server is ready.")
            return

        if server_process.poll() is not None:
            raise RuntimeError(
                "llama-server stopped unexpectedly. Check this log file:\n"
                f"{log_path}"
            )

        time.sleep(1)
        seconds_waited += 1

        if seconds_waited % 30 == 0:
            print("Still waiting for Lynx model server...")


def stop_llama_server() -> None:
    """
    Stop the llama-server process if this Python program started it.
    """
    global server_process
    global server_log_file

    if server_process is not None and server_process.poll() is None:
        print("\nStopping Lynx model server...")
        server_process.terminate()

        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()

    if server_log_file is not None:
        server_log_file.close()
        server_log_file = None
