import json
import os
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="AI Chat", layout="wide")

DB_PATH = "chat_history.db"
DATA_DIR = Path("./data")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_DATA_EXTENSIONS = {".json", ".csv", ".txt", ".pdf"}
MODELS = [
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-plus-02-15",
]
DEFAULT_MODEL = "openai/gpt-5.2"


@st.cache_resource
def get_client() -> OpenAI | None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                seq INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(id)
            )
            """
        )
        conn.commit()


def create_chat(title: str = "New chat") -> dict[str, Any]:
    return {"title": title, "messages": []}


def load_chats_from_db() -> tuple[dict[str, dict[str, Any]], list[str]]:
    chats: dict[str, dict[str, Any]] = {}
    chat_order: list[str] = []

    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, title FROM chats ORDER BY updated_at DESC").fetchall()
        for row in rows:
            chat_id = row["id"]
            chat_order.append(chat_id)
            message_rows = conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY seq ASC",
                (chat_id,),
            ).fetchall()
            chats[chat_id] = {
                "title": row["title"],
                "messages": [
                    {"role": message["role"], "content": message["content"]}
                    for message in message_rows
                ],
            }

    return chats, chat_order


def save_chat_to_db(chat_id: str, chat: dict[str, Any]) -> None:
    now = datetime.now(UTC).isoformat()

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO chats (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                updated_at = excluded.updated_at
            """,
            (chat_id, chat["title"], now, now),
        )

        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))

        for seq, message in enumerate(chat["messages"], start=1):
            conn.execute(
                """
                INSERT INTO messages (chat_id, role, content, seq, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, message["role"], message["content"], seq, now),
            )

        conn.commit()


def build_extra_body(enable_reasoning: bool) -> dict[str, Any]:
    if not enable_reasoning:
        return {}

    return {
        "reasoning": {
            "enabled": True,
            "exclude": False,
        }
    }


def strip_thinking_blocks(text: str) -> tuple[str, str]:
    if not text:
        return "", ""

    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    thinking_text = "\n\n".join(block.strip() for block in think_blocks if block.strip())
    return cleaned_text.strip(), thinking_text


def extract_reasoning_text(reasoning_data: Any) -> str:
    if reasoning_data is None:
        return ""

    if isinstance(reasoning_data, str):
        return reasoning_data

    if isinstance(reasoning_data, list):
        parts: list[str] = []
        for item in reasoning_data:
            nested_text = extract_reasoning_text(item)
            if nested_text:
                parts.append(nested_text)
        return "".join(parts)

    if isinstance(reasoning_data, dict):
        text_value = reasoning_data.get("text")
        if isinstance(text_value, str) and text_value:
            return text_value

        parts: list[str] = []
        for value in reasoning_data.values():
            nested_text = extract_reasoning_text(value)
            if nested_text:
                parts.append(nested_text)
        return "".join(parts)

    return ""


def generate_title_from_first_ai_response(
    client: OpenAI,
    model_name: str,
    response_text: str,
) -> str:
    prompt = (
        "Create a short title of at most 6 words for a conversation based on the following AI response. "
        "Reply with the title only, without quotation marks.\n\n"
        f"{response_text}"
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        title = (completion.choices[0].message.content or "").strip()
        return title[:80] if title else "New chat"
    except Exception:
        fallback = response_text.strip().splitlines()[0] if response_text.strip() else ""
        return fallback[:40] + ("..." if len(fallback) > 40 else "") if fallback else "New chat"


def parse_data_file(file_path: Path) -> dict[str, Any]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        pdf = PdfReader(str(file_path))
        raw_content = "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        raw_content = file_path.read_text(encoding="utf-8")

    if suffix == ".json":
        try:
            parsed_content = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed_content = raw_content
    else:
        parsed_content = raw_content

    return {
        "name": file_path.name,
        "type": suffix,
        "raw": raw_content,
        "parsed": parsed_content,
    }


def load_data_files() -> list[dict[str, Any]]:
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        raise FileNotFoundError("The ./data folder was not found.")

    files = [path for path in sorted(DATA_DIR.iterdir()) if path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS]
    return [parse_data_file(file_path) for file_path in files]


def append_data_to_chat(chat: dict[str, Any], data_items: list[dict[str, Any]]) -> None:
    for item in data_items:
        chat["messages"].append(
            {
                "role": "user",
                "content": f"Data from file {item['name']}:\n{item['raw']}",
            }
        )


def initialize_session_state() -> None:
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL

    if "loaded_data" not in st.session_state:
        st.session_state.loaded_data = []

    if all(key in st.session_state for key in ("chats", "chat_order", "active_chat_id")):
        return

    loaded_chats, loaded_order = load_chats_from_db()
    if loaded_chats:
        st.session_state.chats = loaded_chats
        st.session_state.chat_order = loaded_order
        st.session_state.active_chat_id = loaded_order[0]
        return

    first_chat_id = str(uuid.uuid4())
    st.session_state.chats = {first_chat_id: create_chat()}
    st.session_state.chat_order = [first_chat_id]
    st.session_state.active_chat_id = first_chat_id
    save_chat_to_db(first_chat_id, st.session_state.chats[first_chat_id])


def create_new_chat() -> None:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = create_chat()
    st.session_state.chat_order.insert(0, chat_id)
    st.session_state.active_chat_id = chat_id
    save_chat_to_db(chat_id, st.session_state.chats[chat_id])


def render_sidebar(current_chat: dict[str, Any]) -> bool:
    with st.sidebar:
        st.subheader("⚙️ Settings")

        st.session_state.model_name = st.selectbox(
            "OpenRouter model",
            options=MODELS,
            index=MODELS.index(st.session_state.model_name) if st.session_state.model_name in MODELS else 0,
        )

        enable_reasoning = st.toggle("Enable reasoning", value=True)

        if st.button("+ New chat", use_container_width=True, help="Create a new chat session"):
            create_new_chat()
            st.rerun()

        if st.button(
            "📥 Load data",
            use_container_width=True,
            help="Load .json, .csv, .txt, and .pdf files from the ./data folder",
        ):
            try:
                data_items = load_data_files()
                st.session_state.loaded_data = data_items
                append_data_to_chat(current_chat, data_items)
                save_chat_to_db(st.session_state.active_chat_id, current_chat)
                st.success(f"Loaded {len(data_items)} data file(s).")
                st.rerun()
            except FileNotFoundError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Failed to load data files: {exc}")

        st.divider()
        st.subheader("🗂️ Chat sessions")

        for chat_id in st.session_state.chat_order:
            chat = st.session_state.chats[chat_id]
            is_active = chat_id == st.session_state.active_chat_id
            title = chat["title"] if len(chat["title"]) <= 32 else f"{chat['title'][:32]}..."
            label = f"🟢 {title}" if is_active else title

            if st.button(
                label,
                key=f"chat_{chat_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=f"{len(chat['messages'])} messages",
            ):
                st.session_state.active_chat_id = chat_id
                st.rerun()

    return enable_reasoning


def render_chat_history(chat: dict[str, Any]) -> None:
    for message in chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def stream_assistant_response(
    client: OpenAI,
    model_name: str,
    messages: list[dict[str, str]],
    enable_reasoning: bool,
) -> Iterator[Any]:
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        extra_body=build_extra_body(enable_reasoning),
        stream=True,
    )

    for chunk in completion:
        yield chunk


def handle_user_message(current_chat: dict[str, Any], client: OpenAI | None, enable_reasoning: bool) -> None:
    user_text = st.chat_input("Type your question...")
    if not user_text:
        return

    if client is None:
        st.error("Chat cannot be processed because the OPENROUTER_API_KEY environment variable is missing.")
        st.stop()

    current_chat["messages"].append({"role": "user", "content": user_text})

    with st.chat_message("user"):
        st.markdown(user_text)

    save_chat_to_db(st.session_state.active_chat_id, current_chat)

    with st.chat_message("assistant"):
        reasoning_expander = st.expander("Reasoning", expanded=False)
        reasoning_box = reasoning_expander.empty()
        answer_box = st.empty()
        model_name = st.session_state.get("model_name", DEFAULT_MODEL)

        reasoning_text = ""
        raw_answer_text = ""
        rendered_answer_text = ""

        def response_generator() -> Iterator[str]:
            nonlocal reasoning_text, raw_answer_text, rendered_answer_text

            for chunk in stream_assistant_response(
                client=client,
                model_name=model_name,
                messages=current_chat["messages"],
                enable_reasoning=enable_reasoning,
            ):
                delta = chunk.choices[0].delta
                reasoning_details = getattr(delta, "reasoning_details", None)
                reasoning = getattr(delta, "reasoning", None)
                content = getattr(delta, "content", None)

                streamed_reasoning = extract_reasoning_text(reasoning_details) or extract_reasoning_text(reasoning)
                if streamed_reasoning:
                    reasoning_text += streamed_reasoning
                    reasoning_box.text(reasoning_text.strip())

                if content:
                    raw_answer_text += content
                    cleaned_text, tag_reasoning_text = strip_thinking_blocks(raw_answer_text)

                    visible_reasoning = reasoning_text.strip()
                    if tag_reasoning_text:
                        visible_reasoning = (
                            f"{visible_reasoning}\n\n{tag_reasoning_text}".strip()
                            if visible_reasoning
                            else tag_reasoning_text
                        )

                    if visible_reasoning:
                        reasoning_box.text(visible_reasoning)

                    next_chunk = cleaned_text[len(rendered_answer_text):]
                    if next_chunk:
                        rendered_answer_text = cleaned_text
                        yield next_chunk

        with answer_box:
            streamed_output = st.write_stream(response_generator())

        answer_text = streamed_output if isinstance(streamed_output, str) else rendered_answer_text
        answer_text, tag_reasoning_text = strip_thinking_blocks(answer_text)

        final_reasoning = reasoning_text.strip()
        if tag_reasoning_text:
            final_reasoning = (
                f"{final_reasoning}\n\n{tag_reasoning_text}".strip()
                if final_reasoning
                else tag_reasoning_text
            )

        if final_reasoning:
            reasoning_box.text(final_reasoning)
        else:
            reasoning_expander.empty()

        assistant_count = sum(1 for message in current_chat["messages"] if message["role"] == "assistant")
        current_chat["messages"].append({"role": "assistant", "content": answer_text})

        if assistant_count == 0 and answer_text.strip():
            current_chat["title"] = generate_title_from_first_ai_response(client, model_name, answer_text)

        save_chat_to_db(st.session_state.active_chat_id, current_chat)


init_db()
initialize_session_state()

current_chat = st.session_state.chats[st.session_state.active_chat_id]
client = get_client()
enable_reasoning = render_sidebar(current_chat)

if not current_chat["messages"]:
    st.info("Start the conversation by typing a question below.")

render_chat_history(current_chat)
handle_user_message(current_chat, client, enable_reasoning)
