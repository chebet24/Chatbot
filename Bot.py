"""

Goal:
- Terminal chatbot + Telegram chatbot
- Conversation memory using message arrays
- Persistent personality
- Input validation
- Error handling
- OpenAI model produces replies using the Responses API

Architecture (high level):

User (Terminal / Telegram)
        |
        v
Input validation + session selection (terminal_user or tg_<chat_id>)
        |
        v
Message memory (list of messages per session)
        |
        v
OpenAI call (Responses API) -> assistant reply
        |
        v
Store assistant reply in memory and send to user
"""

# ===============================
# IMPORTS
# ===============================

import os
import sys
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv

# Telegram (async framework)
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# OpenAI official SDK
from openai import OpenAI


# ===============================
# ENV + LOGGING
# ===============================

load_dotenv()  # loads .env into environment variables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("session5_chatbot")


# ===============================
# TYPES FOR MESSAGE ARRAYS
# ===============================

"""
We model conversation memory as a list of Message objects.
This mirrors the common "messages array" structure used in chat systems.

Roles:
- system: sets personality and rules (persistent)
- user: human input
- assistant: bot output
"""
Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


# ===============================
# CHAT ENGINE (BACKEND BRAIN)
# ===============================

@dataclass
class ChatEngine:
    """
    ChatEngine responsibilities:
    1) Keep per-session memory (message arrays)
    2) Validate input (avoid empty/huge messages)
    3) Call OpenAI model to generate replies
    4) Store new messages and trim history for performance

    Important for Telegram:
    - Many users talk to the bot at once
    - We use chat_id as a session key, so each chat has separate memory
    """

    personality: str
    model: str = "gpt-5-mini"  # good balance for teaching cost/speed 2
    max_history: int = 20      # keep last N messages (plus the system message)
    timeout_seconds: int = 30  # simple “sanity” timeout for external calls

    # Stores memory per session:
    # session_id -> [Message(...), Message(...), ...]
    memories: Dict[str, List[Message]] = field(default_factory=dict)

    # One OpenAI client for the whole app (reused across requests)
    client: OpenAI = field(default_factory=lambda: OpenAI())

    def _get_history(self, session_id: str) -> List[Message]:
        """
        Returns the message list for a session.
        If session does not exist, create it and insert the system/personality message first.
        """
        if session_id not in self.memories:
            self.memories[session_id] = [
                Message(role="system", content=self.personality)
            ]
        return self.memories[session_id]

    def _trim_history(self, history: List[Message]) -> None:
        """
        Prevent memory from growing forever.

        WHY?
        - Faster requests
        - Lower token usage (saves money)
        - Avoid huge context windows

        RULE:
        - Always keep the system message at index 0
        - Keep only the last max_history messages after that
        """
        system = history[:1]     # first message only
        rest = history[1:]       # everything after system

        if len(rest) > self.max_history:
            rest = rest[-self.max_history:]  # keep last N

        history[:] = system + rest

    def validate_user_input(self, text: Optional[str]) -> str:
        """
        Basic validation:
        - must exist
        - not empty
        - not too long
        """
        if text is None:
            raise ValueError("Input is missing.")

        cleaned = text.strip()

        if not cleaned:
            raise ValueError("Please type something (not empty).")

        # keep a reasonable limit for class demos
        if len(cleaned) > 2000:
            raise ValueError("Message too long. Keep it under 2000 characters.")

        return cleaned

    def remember(self, session_id: str, role: Role, content: str) -> None:
        """
        Append a new message into the message array for that session,
        then trim to keep memory under control.
        """
        history = self._get_history(session_id)
        history.append(Message(role=role, content=content))
        self._trim_history(history)

    def _messages_as_openai_input(self, history: List[Message]) -> List[dict]:
        """
        Convert our Message dataclasses into the dict format
        expected by OpenAI chat-style input.

        We keep it explicit for teaching clarity.
        """
        return [{"role": m.role, "content": m.content} for m in history]

    def generate_reply_openai(self, session_id: str) -> str:
        """
        This is where the OpenAI model is called.

        We DO NOT take user_text here because we already stored it in memory.
        The full conversation history becomes our prompt context.

        Using the Responses API lets us produce a model response from input.
        Docs: Responses API reference. 3
        """
        history = self._get_history(session_id)

        # Convert message array into OpenAI input format
        input_messages = self._messages_as_openai_input(history)

        # IMPORTANT TEACHING NOTE:
        # - The system message stays at the start and enforces the personality.
        # - The conversation memory provides context for follow-up questions.

        try:
            # Create a response from the model
            # "input" can be message arrays for multi-turn context
            response = self.client.responses.create(
                model=self.model,
                input=input_messages,
            )

            # SDK convenience: response.output_text contains aggregated text output (when available).
            # This is noted in the API reference. 4
            text = (response.output_text or "").strip()

            if not text:
                # Defensive fallback — if the model returns empty text
                return "I didn’t get that. Could you rephrase?"

            return text

        except Exception as e:
            # Log full details for developer debugging
            logger.exception("OpenAI call failed: %s", e)

            # Show user-friendly message (no stack trace)
            return "❌ I had trouble generating a reply just now. Try again."

    # --- Async helper for Telegram (avoid blocking event loop) ---
    async def generate_reply_openai_async(self, session_id: str) -> str:
        """
        Telegram handlers are async.

        The OpenAI SDK call is synchronous.
        If we call it directly, it blocks the async event loop (bad for multi-user).

        asyncio.to_thread(...) runs the blocking OpenAI call in a thread.
        That keeps Telegram responsive.
        """
        return await asyncio.to_thread(self.generate_reply_openai, session_id)


# ===============================
# TERMINAL CHAT LOOP
# ===============================

async def run_terminal(engine: ChatEngine) -> None:
    """
    Terminal mode: single “session_id”.
    Great for live demo because students can run it without Telegram.
    """
    session_id = "terminal_user"

    print("Terminal Chatbot (OpenAI). Type /quit to exit.\n")

    while True:
        try:
            user_text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if user_text.strip().lower() in {"/quit", "/exit"}:
            print("Bye 👋")
            return

        try:
            cleaned = engine.validate_user_input(user_text)

            # Store user message in memory BEFORE calling model
            engine.remember(session_id, "user", cleaned)

            # Call OpenAI to generate a reply using full conversation history
            reply = engine.generate_reply_openai(session_id)

            # Store assistant reply in memory
            engine.remember(session_id, "assistant", reply)

            print(f"Bot: {reply}\n")

        except ValueError as ve:
            print(f"Bot: ⚠️ {ve}\n")

        except Exception:
            logger.exception("Unexpected terminal error")
            print("Bot: ❌ Something went wrong. Try again.\n")


# ===============================
# TELEGRAM HELPERS + HANDLERS
# ===============================

def telegram_session_id(update: Update) -> str:
    """
    Telegram has many chats/users.
    Use chat_id to separate memory per chat:
    memory["tg_12345"] is different from memory["tg_77777"].
    """
    chat_id = update.effective_chat.id if update.effective_chat else "unknown"
    return f"tg_{chat_id}"


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    engine: ChatEngine = context.application.bot_data["engine"]
    sid = telegram_session_id(update)

    msg = "Hi! I’m your OpenAI chatbot. Send a message and I’ll remember context."
    engine.remember(sid, "assistant", msg)
    await update.message.reply_text(msg)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    engine: ChatEngine = context.application.bot_data["engine"]
    sid = telegram_session_id(update)

    try:
        user_text = update.message.text if update.message else ""
        cleaned = engine.validate_user_input(user_text)

        # Store user message
        engine.remember(sid, "user", cleaned)

        # Generate reply in a thread (non-blocking for async Telegram)
        reply = await engine.generate_reply_openai_async(sid)

        # Store assistant message
        engine.remember(sid, "assistant", reply)

        await update.message.reply_text(reply)

    except ValueError as ve:
        await update.message.reply_text(f"⚠️ {ve}")

    except Exception:
        logger.exception("Telegram handler error")
        if update.message:
            await update.message.reply_text("❌ Something went wrong. Please try again.")


async def run_telegram(engine: ChatEngine) -> None:
    """
    Runs the Telegram bot via long polling.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env or environment variables.")

    app = Application.builder().token(token).build()

    # Share the engine (and its memory) with all handlers
    app.bot_data["engine"] = engine

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Telegram bot running…")
    await app.run_polling(close_loop=False)


# ===============================
# ENGINE FACTORY
# ===============================

def build_engine() -> ChatEngine:
    """
    Define the bot personality here (persistent).
    This becomes the system message that remains at the start of memory.
    """
    personality = (
        "You are a friendly programming tutor chatbot. "
        "You answer clearly, in short steps, and ask a follow-up question when helpful. "
        "If the user asks for code, provide clean examples. "
        "If the user input is unclear, ask for clarification."
    )

    # Choose a model:
    
    return ChatEngine(personality=personality, model=os.getenv("OPENAI_MODEL", "gpt-5-mini"))


# ===============================
# MAIN ENTRYPOINT
# ===============================

if __name__ == "__main__":
    """
    Run modes:

    Terminal (default):
        python bot.py

    Telegram:
        python bot.py telegram
    """
    # Ensure OpenAI key is set early so students see errors clearly
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env or environment variables.") 

    engine = build_engine()

    mode = "terminal"
    if len(sys.argv) >= 2:
        mode = sys.argv[1].strip().lower()

    if mode == "telegram":
        asyncio.run(run_telegram(engine))
    else:
        asyncio.run(run_terminal(engine))
