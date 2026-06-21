"""
active_memory.py

Temporary active memory for the current Lynx chat session.

This module does NOT save anything permanently.
It only keeps recent messages in RAM while the program is running.
"""


class ActiveMemory:
    def __init__(self, system_prompt: str, max_messages: int = 30):
        """
        Create active memory for one chat session.

        system_prompt:
            The permanent instruction message given to the model.

        max_messages:
            Maximum number of recent non-system messages to keep.
            Example: 30 means roughly 15 user messages + 15 assistant replies.
        """

        self.system_prompt = system_prompt
        self.max_messages = max_messages

        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

    def add_user_message(self, content: str) -> None:
        """Add a user message to active memory."""
        self.messages.append(
            {
                "role": "user",
                "content": content,
            }
        )
        self._trim_if_needed()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant/model message to active memory."""
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
            }
        )
        self._trim_if_needed()

    def get_messages(self) -> list[dict[str, str]]:
        """
        Return the current message history.

        A copy is returned so other modules cannot accidentally damage
        the internal memory list.
        """
        return list(self.messages)

    def clear(self) -> None:
        """
        Clear active memory, but keep the system prompt.
        """
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

    def _trim_if_needed(self) -> None:
        """
        Keep active memory from growing forever.

        The system message is always preserved.
        Only old user/assistant messages are removed.
        """

        system_message = self.messages[0]
        conversation_messages = self.messages[1:]

        if len(conversation_messages) > self.max_messages:
            conversation_messages = conversation_messages[-self.max_messages:]

        self.messages = [system_message] + conversation_messages

    def message_count(self) -> int:
        """
        Return number of non-system messages in active memory.
        """
        return len(self.messages) - 1
