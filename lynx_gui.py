"""
lynx_gui.py

Minimal desktop GUI for Lynx.

This file is intentionally separate from lynx.py.
It provides a simple PySide6 desktop interface with:
- scrolling message screen
- input box
- Send button
- Kill button

It uses existing lynx_core modules:
- model_server.py
- active_memory.py
- database.py
- memory_retriever.py
- conversation_summarizer.py
"""

import sys
import requests

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

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
from lynx_core.wikipedia_tool import (
    extract_wikipedia_query,
    format_wikipedia_context,
    lookup_wikipedia_summary,
)


SYSTEM_PROMPT = (
    "You are Lynx, a local AI assistant. "
    "Answer clearly, directly, and helpfully."
)


def ask_model(messages: list[dict[str, str]]) -> str:
    """
    Send messages to the local llama.cpp server and return Lynx's answer.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.4,
    }

    response = requests.post(CHAT_URL, json=payload, timeout=None)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def build_messages_for_model(
    active_messages: list[dict[str, str]],
    recalled_context: str,
    wikipedia_context: str = "",
) -> list[dict[str, str]]:
    """
    Insert retrieved memory and Wikipedia context into the prompt
    without polluting active memory.
    """

    messages_for_model = list(active_messages)
    insert_position = 1

    if wikipedia_context:
        wikipedia_message = {
            "role": "system",
            "content": (
                "Wikipedia information retrieved for the user's current request:"
                f"{wikipedia_context}"
                "Use this source information when answering the user's current message. "
                "Do not claim the information came from memory; it came from Wikipedia."
            ),
        }
        messages_for_model.insert(insert_position, wikipedia_message)
        insert_position += 1

    if recalled_context:
        memory_message = {
            "role": "system",
            "content": (
                "Relevant memory retrieved from previous conversations:"
                f"{recalled_context}"
                "Use this memory only if it helps answer the user's current message."
            ),
        }
        messages_for_model.insert(insert_position, memory_message)

    return messages_for_model


class StartServerWorker(QObject):
    """
    Starts llama-server in a background thread so the GUI does not freeze.
    """

    finished = Signal()
    error = Signal(str)

    def run(self) -> None:
        try:
            start_llama_server()
            self.finished.emit()
        except Exception as error:
            self.error.emit(str(error))


class SendMessageWorker(QObject):
    """
    Handles one user message in a background thread.

    This worker uses its own database connection for memory retrieval.
    The main GUI thread saves user/assistant messages to the current conversation.
    """

    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        user_message: str,
        active_messages: list[dict[str, str]],
        conversation_id: int,
    ):
        super().__init__()
        self.user_message = user_message
        self.active_messages = active_messages
        self.conversation_id = conversation_id

    def run(self) -> None:
        try:
            # Use a separate SQLite connection in this worker thread.
            db = LynxDatabase()

            wikipedia_context = ""
            wikipedia_query = extract_wikipedia_query(self.user_message)

            if wikipedia_query:
                wikipedia_result = lookup_wikipedia_summary(wikipedia_query)

                if wikipedia_result is None:
                    wikipedia_context = (
                        f"No Wikipedia article summary was found for: {wikipedia_query}"
                    )
                else:
                    wikipedia_context = format_wikipedia_context(wikipedia_result)

                    db.save_web_source(
                        source_type="wikipedia",
                        title=wikipedia_result.title,
                        url=wikipedia_result.url,
                        query=wikipedia_result.query,
                        summary=wikipedia_result.summary,
                        conversation_id=self.conversation_id,
                    )

            recalled_context = retrieve_relevant_context(
                db=db,
                user_message=self.user_message,
                summary_limit=30,
            )

            db.close()

            messages_for_model = build_messages_for_model(
                active_messages=self.active_messages,
                recalled_context=recalled_context,
                wikipedia_context=wikipedia_context,
            )

            answer = ask_model(messages_for_model)
            self.finished.emit(answer)

        except Exception as error:
            self.error.emit(str(error))


class ShutdownWorker(QObject):
    """
    Summarizes the current session before shutdown.
    """

    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        conversation_id: int,
        session_messages_for_summary: list[dict[str, str]],
    ):
        super().__init__()
        self.conversation_id = conversation_id
        self.session_messages_for_summary = session_messages_for_summary

    def run(self) -> None:
        try:
            # Use a separate SQLite connection in this worker thread.
            db = LynxDatabase()

            summary = summarize_and_store_current_conversation(
                db=db,
                conversation_id=self.conversation_id,
                active_messages=self.session_messages_for_summary,
                max_summary_tokens=250,
            )

            db.end_conversation(self.conversation_id)
            db.close()

            self.finished.emit(summary)

        except Exception as error:
            self.error.emit(str(error))


class LynxMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Lynx")
        self.resize(900, 650)

        self.db = LynxDatabase()
        self.conversation_id = self.db.start_conversation(title="Lynx GUI chat")

        self.memory = ActiveMemory(
            system_prompt=SYSTEM_PROMPT,
            max_messages=30,
        )

        # This transcript is used only for shutdown summary.
        self.session_messages_for_summary: list[dict[str, str]] = []

        self.worker_thread: QThread | None = None
        self.worker: QObject | None = None
        self.shutting_down = False

        self.chat_screen = QTextEdit()
        self.chat_screen.setReadOnly(True)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type your message here...")
        self.input_box.returnPressed.connect(self.send_message)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)

        self.kill_button = QPushButton("Kill")
        self.kill_button.clicked.connect(self.kill_lynx)

        self.status_label = QLabel("Starting Lynx model server...")

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.send_button)
        input_layout.addWidget(self.kill_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.chat_screen)
        main_layout.addLayout(input_layout)
        main_layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.set_input_enabled(False)
        self.append_system_message("Starting Lynx model server. This may take a while...")

        self.start_server_in_background()

    def append_system_message(self, text: str) -> None:
        self.chat_screen.append(f"<b>System:</b> {self.escape_html(text)}")
        self.scroll_to_bottom()

    def append_user_message(self, text: str) -> None:
        self.chat_screen.append(f"<b>You:</b> {self.escape_html(text)}")
        self.scroll_to_bottom()

    def append_lynx_message(self, text: str) -> None:
        safe_text = self.escape_html(text).replace("\n", "<br>")
        self.chat_screen.append(f"<b>Lynx:</b> {safe_text}")
        self.scroll_to_bottom()

    def append_error_message(self, text: str) -> None:
        self.chat_screen.append(f"<b>Error:</b> {self.escape_html(text)}")
        self.scroll_to_bottom()

    def scroll_to_bottom(self) -> None:
        scrollbar = self.chat_screen.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_input_enabled(self, enabled: bool) -> None:
        self.input_box.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.kill_button.setEnabled(True)

    @staticmethod
    def escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def start_server_in_background(self) -> None:
        self.worker_thread = QThread()
        self.worker = StartServerWorker()
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_server_ready)
        self.worker.error.connect(self.on_server_error)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)

        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def on_server_ready(self) -> None:
        self.status_label.setText("Lynx model server is ready.")
        self.append_system_message("Lynx is ready.")
        self.set_input_enabled(True)
        self.input_box.setFocus()

    def on_server_error(self, error: str) -> None:
        self.status_label.setText("Failed to start Lynx model server.")
        self.append_error_message(error)
        self.set_input_enabled(False)

    def send_message(self) -> None:
        if self.shutting_down:
            return

        user_message = self.input_box.text().strip()

        if not user_message:
            return

        self.input_box.clear()
        self.set_input_enabled(False)
        self.status_label.setText("Lynx is thinking...")

        self.append_user_message(user_message)

        self.memory.add_user_message(user_message)
        self.session_messages_for_summary.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        try:
            self.db.save_message(self.conversation_id, "user", user_message)
        except Exception as error:
            self.append_error_message(f"Could not save user message: {error}")

        active_messages = self.memory.get_messages()

        self.worker_thread = QThread()
        self.worker = SendMessageWorker(
            user_message=user_message,
            active_messages=active_messages,
            conversation_id=self.conversation_id,
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_answer_ready)
        self.worker.error.connect(self.on_answer_error)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)

        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def on_answer_ready(self, answer: str) -> None:
        self.memory.add_assistant_message(answer)
        self.session_messages_for_summary.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        try:
            self.db.save_message(self.conversation_id, "assistant", answer)
        except Exception as error:
            self.append_error_message(f"Could not save assistant message: {error}")

        self.append_lynx_message(answer)
        self.status_label.setText("Ready.")
        self.set_input_enabled(True)
        self.input_box.setFocus()

    def on_answer_error(self, error: str) -> None:
        self.append_error_message(error)
        self.status_label.setText("Error.")
        self.set_input_enabled(True)
        self.input_box.setFocus()

    def kill_lynx(self) -> None:
        if self.shutting_down:
            return

        self.shutting_down = True
        self.set_input_enabled(False)
        self.kill_button.setEnabled(False)

        self.status_label.setText("Summarizing current conversation before shutdown...")
        self.append_system_message("Summarizing current conversation before shutdown...")

        self.worker_thread = QThread()
        self.worker = ShutdownWorker(
            conversation_id=self.conversation_id,
            session_messages_for_summary=self.session_messages_for_summary,
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_shutdown_summary_ready)
        self.worker.error.connect(self.on_shutdown_error)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)

        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def on_shutdown_summary_ready(self, summary: str) -> None:
        if summary:
            self.append_system_message("Conversation summary saved.")
            self.append_system_message(summary)
        else:
            self.append_system_message("Conversation summary step completed.")

        self.finish_shutdown()

    def on_shutdown_error(self, error: str) -> None:
        self.append_error_message(f"Could not summarize conversation: {error}")
        self.finish_shutdown()

    def finish_shutdown(self) -> None:
        self.status_label.setText("Stopping Lynx model server...")

        try:
            self.db.close()
        except Exception:
            pass

        try:
            stop_llama_server()
        except Exception as error:
            self.append_error_message(f"Could not stop model server cleanly: {error}")

        QApplication.quit()

    def closeEvent(self, event) -> None:
        """
        Prevent accidental window closing.

        The user must use the Kill button so Lynx can summarize the session
        and stop the model server cleanly.
        """
        if self.shutting_down:
            event.accept()
            return

        event.ignore()
        self.append_system_message("Use the Kill button to close Lynx safely.")


def main() -> None:
    app = QApplication(sys.argv)
    window = LynxMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
