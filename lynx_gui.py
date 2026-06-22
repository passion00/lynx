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
- wikipedia_tool.py
- fact_extractor.py
- file_tools.py
"""

import sys
import requests

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lynx_core.active_memory import ActiveMemory
from lynx_core.database import LynxDatabase
from lynx_core.memory_retriever import retrieve_relevant_context
from lynx_core.conversation_summarizer import summarize_and_store_current_conversation
from lynx_core.fact_extractor import extract_and_store_facts
from lynx_core.settings import LynxSettings, load_settings, save_settings, server_restart_required
from lynx_core.model_server import (
    get_chat_url,
    start_llama_server,
    stop_llama_server,
)
from lynx_core.wikipedia_tool import (
    extract_wikipedia_query,
    format_wikipedia_context,
    lookup_wikipedia_summary,
)
from lynx_core.file_tool_router import run_autonomous_file_tool


SYSTEM_PROMPT = (
    "You are Lynx, a local AI assistant. "
    "Answer clearly, directly, and helpfully."
)


def ask_model(messages: list[dict[str, str]]) -> str:
    """
    Send messages to the local llama.cpp server and return Lynx's answer.
    """

    settings = load_settings()

    payload = {
        "model": settings.model_name,
        "messages": messages,
        "temperature": settings.temperature,
    }

    if settings.max_response_tokens > 0:
        payload["max_tokens"] = settings.max_response_tokens

    response = requests.post(get_chat_url(settings), json=payload, timeout=None)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def build_messages_for_model(
    active_messages: list[dict[str, str]],
    recalled_context: str,
    wikipedia_context: str = "",
    file_context: str = "",
) -> list[dict[str, str]]:
    """
    Insert retrieved memory, Wikipedia context, and file-tool context into the prompt
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

    if file_context:
        file_message = {
            "role": "system",
            "content": (
                "Filesystem inspection result for the user's current request:"
                f"{file_context}"
                "Use this file information when answering the user's current message. "
                "Do not claim the information came from memory; it came from a read-only file tool."
            ),
        }
        messages_for_model.insert(insert_position, file_message)
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


class ModelSettingsDialog(QDialog):
    """Small GUI dialog for editing Lynx model settings."""

    def __init__(self, settings: LynxSettings, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Lynx Model Settings")
        self.setModal(True)

        self.model_name_input = QLineEdit(settings.model_name)

        self.context_size_input = QSpinBox()
        self.context_size_input.setRange(512, 131072)
        self.context_size_input.setSingleStep(512)
        self.context_size_input.setValue(settings.context_size)

        self.gpu_layers_input = QSpinBox()
        self.gpu_layers_input.setRange(0, 999)
        self.gpu_layers_input.setValue(settings.gpu_layers)

        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setRange(0.0, 2.0)
        self.temperature_input.setDecimals(2)
        self.temperature_input.setSingleStep(0.05)
        self.temperature_input.setValue(settings.temperature)

        self.max_response_tokens_input = QSpinBox()
        self.max_response_tokens_input.setRange(0, 32768)
        self.max_response_tokens_input.setSingleStep(128)
        self.max_response_tokens_input.setValue(settings.max_response_tokens)
        self.max_response_tokens_input.setSpecialValueText("Default / unlimited")

        self.fact_extractor_tokens_input = QSpinBox()
        self.fact_extractor_tokens_input.setRange(32, 2048)
        self.fact_extractor_tokens_input.setSingleStep(32)
        self.fact_extractor_tokens_input.setValue(settings.fact_extractor_max_tokens)

        self.host_input = QLineEdit(settings.host)

        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(settings.port)

        form_layout = QFormLayout()
        form_layout.addRow("Model repo/tag:", self.model_name_input)
        form_layout.addRow("Context size (-c):", self.context_size_input)
        form_layout.addRow("GPU layers (-ngl):", self.gpu_layers_input)
        form_layout.addRow("Temperature:", self.temperature_input)
        form_layout.addRow("Max response tokens:", self.max_response_tokens_input)
        form_layout.addRow("Fact extractor tokens:", self.fact_extractor_tokens_input)
        form_layout.addRow("Host:", self.host_input)
        form_layout.addRow("Port:", self.port_input)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def get_settings(self) -> LynxSettings:
        settings = LynxSettings(
            model_name=self.model_name_input.text().strip(),
            context_size=self.context_size_input.value(),
            gpu_layers=self.gpu_layers_input.value(),
            temperature=self.temperature_input.value(),
            max_response_tokens=self.max_response_tokens_input.value(),
            fact_extractor_max_tokens=self.fact_extractor_tokens_input.value(),
            host=self.host_input.text().strip(),
            port=self.port_input.value(),
        )
        settings.normalize()
        return settings


class RestartServerWorker(QObject):
    """Stops and starts llama-server after model/server settings change."""

    finished = Signal()
    error = Signal(str)

    def run(self) -> None:
        try:
            stop_llama_server()
            start_llama_server()
            self.finished.emit()
        except Exception as error:
            self.error.emit(str(error))


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

    Order of operations:
    1. Extract durable facts from the latest user message and save them.
    2. Retrieve relevant memory, checking facts first.
    3. Let the autonomous file-tool router decide whether filesystem inspection is needed.
    4. Fetch Wikipedia context if explicitly requested.
    5. Ask the model for the assistant response.
    """

    finished = Signal(str, int, str)
    error = Signal(str)

    def __init__(
        self,
        user_message: str,
        active_messages: list[dict[str, str]],
        conversation_id: int,
        current_message_id: int | None,
    ):
        super().__init__()
        self.user_message = user_message
        self.active_messages = active_messages
        self.conversation_id = conversation_id
        self.current_message_id = current_message_id

    def run(self) -> None:
        try:
            # Use a separate SQLite connection in this worker thread.
            db = LynxDatabase()

            extracted_facts = extract_and_store_facts(
                db=db,
                user_message=self.user_message,
                conversation_id=self.conversation_id,
                source_message_id=self.current_message_id,
            )

            file_context = ""
            file_tool_title = ""
            file_tool_result = run_autonomous_file_tool(self.user_message)
            if file_tool_result is not None:
                file_tool_title = file_tool_result.title
                file_context = file_tool_result.content

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
                current_message_id=self.current_message_id,
            )

            db.close()

            messages_for_model = build_messages_for_model(
                active_messages=self.active_messages,
                recalled_context=recalled_context,
                wikipedia_context=wikipedia_context,
                file_context=file_context,
            )

            answer = ask_model(messages_for_model)
            self.finished.emit(answer, len(extracted_facts), file_tool_title)

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

        self.create_menu_bar()

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

    def create_menu_bar(self) -> None:
        settings_menu = self.menuBar().addMenu("Settings")

        model_settings_action = QAction("Model settings...", self)
        model_settings_action.triggered.connect(self.open_model_settings_dialog)
        settings_menu.addAction(model_settings_action)

        tools_menu = self.menuBar().addMenu("Tools")
        file_help_action = QAction("Autonomous file tools", self)
        file_help_action.triggered.connect(self.show_file_tool_help)
        tools_menu.addAction(file_help_action)

    def show_file_tool_help(self) -> None:
        self.append_system_message(
            "Autonomous file tools: read file: /path/to/file | inspect file: /path/to/file | "
            "list directory: /path/to/folder | ls: /path/to/folder | "
            "search files: pattern in /path/to/root | search text: query in /path/to/root | "
            "write playground: relative/path.txt <<< content. "
            "File tools are read-only outside ~/lynx/playground. Playground writes create new files only and refuse overwrites."
        )

    def open_model_settings_dialog(self) -> None:
        if self.shutting_down:
            return

        old_settings = load_settings()
        dialog = ModelSettingsDialog(old_settings, self)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        new_settings = dialog.get_settings()
        needs_restart = server_restart_required(old_settings, new_settings)
        save_settings(new_settings)

        if needs_restart:
            self.append_system_message(
                "Settings saved. Restarting model server so the new model/server settings apply."
            )
            self.restart_server_in_background()
        else:
            self.append_system_message("Settings saved. New chat settings will apply to the next answer.")

    def restart_server_in_background(self) -> None:
        self.set_input_enabled(False)
        self.status_label.setText("Restarting Lynx model server...")

        self.worker_thread = QThread()
        self.worker = RestartServerWorker()
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_server_restarted)
        self.worker.error.connect(self.on_server_restart_error)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)

        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def on_server_restarted(self) -> None:
        self.status_label.setText("Lynx model server restarted.")
        self.append_system_message("Model server restarted with the saved settings.")
        self.set_input_enabled(True)
        self.input_box.setFocus()

    def on_server_restart_error(self, error: str) -> None:
        self.status_label.setText("Failed to restart Lynx model server.")
        self.append_error_message(error)
        self.set_input_enabled(False)

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
        self.status_label.setText("Lynx is extracting facts, checking tools, and thinking...")

        self.append_user_message(user_message)

        self.memory.add_user_message(user_message)
        self.session_messages_for_summary.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        user_message_id: int | None = None

        try:
            user_message_id = self.db.save_message(
                self.conversation_id,
                "user",
                user_message,
            )
        except Exception as error:
            self.append_error_message(f"Could not save user message: {error}")

        active_messages = self.memory.get_messages()

        self.worker_thread = QThread()
        self.worker = SendMessageWorker(
            user_message=user_message,
            active_messages=active_messages,
            conversation_id=self.conversation_id,
            current_message_id=user_message_id,
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

    def on_answer_ready(self, answer: str, facts_extracted_count: int = 0, file_tool_title: str = "") -> None:
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

        if facts_extracted_count > 0:
            self.append_system_message(
                f"Saved {facts_extracted_count} durable fact(s) to memory."
            )

        if file_tool_title:
            self.append_system_message(f"File tool used: {file_tool_title}.")

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
