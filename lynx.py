import atexit
import sys

import requests

from lynx_core.active_memory import ActiveMemory
from lynx_core.database import LynxDatabase
from lynx_core.memory_retriever import retrieve_relevant_context
from lynx_core.conversation_summarizer import summarize_and_store_current_conversation
from lynx_core.model_server import (
    CHAT_URL,
    MODEL_NAME,
    start_llama_server,
    stop_llama_server,
)


# ===== Lynx configuration =====

SYSTEM_PROMPT = (
    "You are Lynx, a local AI assistant. "
    "Answer clearly, directly, and helpfully."
)

TEMPERATURE = 0.4
ACTIVE_MEMORY_MAX_MESSAGES = 30
SUMMARY_RETRIEVAL_LIMIT = 30
SHUTDOWN_SUMMARY_MAX_TOKENS = 250


def ask_model(messages: list[dict[str, str]]) -> str:
    """
    Send chat messages to the local model server and return the answer.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
    }

    response = requests.post(CHAT_URL, json=payload, timeout=None)
    response.raise_for_status()

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as error:
        raise RuntimeError(f"Unexpected model response format: {data}") from error


def read_user_input(prompt: str = "\nYou: ") -> str:
    """
    Read user input safely.

    This prevents Lynx from crashing if the terminal sends malformed UTF-8 bytes.
    """

    print(prompt, end="", flush=True)

    raw_input = sys.stdin.buffer.readline()

    if raw_input == b"":
        return "kill"

    return raw_input.decode("utf-8", errors="replace").strip()


def build_messages_for_model(
    memory: ActiveMemory,
    recalled_context: str,
) -> list[dict[str, str]]:
    """
    Build the message list sent to the model.

    Retrieved database memory is inserted only into this temporary message list.
    Active memory remains clean.
    """

    messages_for_model = memory.get_messages()

    if recalled_context:
        memory_message = {
            "role": "system",
            "content": (
                "Relevant memory retrieved from previous conversations:\n\n"
                f"{recalled_context}\n\n"
                "Use this memory only if it helps answer the user's current message. "
                "Do not mention the retrieval process unless the user asks about it."
            ),
        }

        # Insert after the main system prompt.
        messages_for_model.insert(1, memory_message)

    return messages_for_model


def main() -> None:
    atexit.register(stop_llama_server)

    db = None
    conversation_id = None

    # Independent transcript for shutdown summarization.
    # This does not depend on active memory, trimming, or database retrieval.
    session_messages_for_summary: list[dict[str, str]] = []

    try:
        start_llama_server()

        memory = ActiveMemory(
            system_prompt=SYSTEM_PROMPT,
            max_messages=ACTIVE_MEMORY_MAX_MESSAGES,
        )

        db = LynxDatabase()
        conversation_id = db.start_conversation(title="Lynx chat")

        print("\nLynx controller started.")
        print('Type "kill" to exit.')
        print('Type "clear" to clear active memory.')
        print('Type "status" to show active memory status.')

        while True:
            user_input = read_user_input()
            command = user_input.lower()

            if command == "kill":
                print("Summarizing current conversation before shutdown...")
                print(
                    f"Shutdown summary transcript contains "
                    f"{len(session_messages_for_summary)} messages."
                )

                try:
                    summary = summarize_and_store_current_conversation(
                        db=db,
                        conversation_id=conversation_id,
                        active_messages=session_messages_for_summary,
                        max_summary_tokens=SHUTDOWN_SUMMARY_MAX_TOKENS,
                    )

                    if summary:
                        print("\nConversation summary saved:")
                        print(summary)
                    else:
                        print("No conversation content to summarize.")

                except Exception as error:
                    print(f"\nWarning: Could not summarize conversation: {error}")

                print("Lynx controller stopped.")
                break

            if command == "clear":
                memory.clear()
                print("Active memory cleared.")
                continue

            if command == "status":
                print(f"Active memory contains {memory.message_count()} messages.")
                print(f"Shutdown summary transcript contains {len(session_messages_for_summary)} messages.")
                print(f"Current conversation ID: {conversation_id}")
                print(f"Database path: {db.db_path}")
                continue

            if not user_input:
                continue

            try:
                # Save user message in all three places:
                # 1. active memory for current-context chat
                # 2. SQLite for permanent raw archive
                # 3. session transcript for shutdown summary
                memory.add_user_message(user_input)
                db.save_message(conversation_id, "user", user_input)
                session_messages_for_summary.append(
                    {"role": "user", "content": user_input}
                )

                recalled_context = retrieve_relevant_context(
                    db=db,
                    user_message=user_input,
                    summary_limit=SUMMARY_RETRIEVAL_LIMIT,
                )

                messages_for_model = build_messages_for_model(
                    memory=memory,
                    recalled_context=recalled_context,
                )

                answer = ask_model(messages_for_model)

                # Save assistant message in all three places.
                memory.add_assistant_message(answer)
                db.save_message(conversation_id, "assistant", answer)
                session_messages_for_summary.append(
                    {"role": "assistant", "content": answer}
                )

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
