import os
import json
import re
import sqlite3
import uuid
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import tiktoken
from datetime import UTC, datetime
from pathlib import Path

# --- Load .env ---
load_dotenv()

# --- Page setup (wide, no title) ---
st.set_page_config(page_title="Chat AI", layout="wide")

st.markdown(
    """
    <style>
        .chat-tip {
            border: 1px dashed rgba(148, 163, 184, 0.45);
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0 1rem 0;
            color: #cbd5e1;
        }
        .stSidebar .stButton button {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Fixed rules ---
CHUNK_SIZE_TOKENS = 8192
OVERLAP_TOKENS = 100
TIKTOKEN_ENCODING = "cl100k_base"

DB_PATH = "chat_history.db"
DATA_DIR = Path("./data")

# --- Provider & model options ---
PROVIDERS = {
    "DeepSeek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-chat"],
    },
    "DashScope (Qwen)": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen3.5-plus"],
    },
}

DEFAULT_PROVIDER = "DeepSeek"
DEFAULT_MODEL_BY_PROVIDER = {
    "DeepSeek": "deepseek-chat",
    "DashScope (Qwen)": "qwen3.5-plus",
}


@st.cache_resource
def get_client(provider_name: str):
    provider = PROVIDERS.get(provider_name)
    if not provider:
        return None

    api_key = os.getenv(provider["env_key"])
    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url=provider["base_url"],
    )


@st.cache_resource
def get_encoder():
    return tiktoken.get_encoding(TIKTOKEN_ENCODING)


def count_tokens(text: str) -> int:
    enc = get_encoder()
    return len(enc.encode(text))


def count_total_session_chunks(messages: list[dict]) -> int:
    """
    Hitung total chunk token berdasarkan seluruh isi session messages.
    1 pesan = ceil(total_token_pesan / CHUNK_SIZE_TOKENS), minimal 1 chunk bila ada isi.
    """
    total_chunks = 0
    for message in messages:
        content = (message.get("content") or "").strip()
        if not content:
            continue

        token_count = count_tokens(content)
        total_chunks += max(1, (token_count + CHUNK_SIZE_TOKENS - 1) // CHUNK_SIZE_TOKENS)

    return total_chunks


def split_with_overlap_8192_100(text: str) -> list[str]:
    enc = get_encoder()
    toks = enc.encode(text)
    if not toks:
        return [""]

    step = CHUNK_SIZE_TOKENS - OVERLAP_TOKENS
    chunks = []
    for start in range(0, len(toks), step):
        end = min(start + CHUNK_SIZE_TOKENS, len(toks))
        chunks.append(enc.decode(toks[start:end]))
        if end == len(toks):
            break
    return chunks


def build_extra_body(provider_name: str, enable_thinking: bool) -> dict:
    """
    DashScope/Qwen: extra_body={"enable_thinking": True/False}
    DeepSeek: extra_body={"thinking": {"type": "enabled"/"disabled"}}
    """
    if provider_name == "DashScope (Qwen)":
        return {"enable_thinking": enable_thinking}
    return {"thinking": {"type": "enabled" if enable_thinking else "disabled"}}


def strip_thinking_blocks(text: str) -> tuple[str, str]:
    """
    Pisahkan blok <think>...</think> dari konten jawaban utama.
    Return: (jawaban_bersih, thinking_dari_tag)
    """
    if not text:
        return "", ""

    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip(), "\n\n".join(block.strip() for block in think_blocks if block.strip())


def create_chat(title: str = "New chat") -> dict:
    return {"title": title, "messages": []}


def get_chat_title(messages: list[dict]) -> str:
    for message in messages:
        if message["role"] == "user" and message["content"].strip():
            content = message["content"].strip().replace("\n", " ")
            return content[:40] + ("..." if len(content) > 40 else "")
    return "New chat"


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


def load_chats_from_db() -> tuple[dict, list]:
    chats: dict[str, dict] = {}
    chat_order: list[str] = []

    with get_db_connection() as conn:
        chat_rows = conn.execute(
            "SELECT id, title FROM chats ORDER BY updated_at DESC"
        ).fetchall()

        for row in chat_rows:
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


def save_chat_to_db(chat_id: str, chat: dict) -> None:
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
        for idx, message in enumerate(chat["messages"], start=1):
            conn.execute(
                """
                INSERT INTO messages (chat_id, role, content, seq, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, message["role"], message["content"], idx, now),
            )
        conn.commit()


def generate_title_from_first_ai_response(
    client: OpenAI,
    provider_name: str,
    model_name: str,
    response_text: str,
) -> str:
    prompt = (
        "Buat judul singkat (maksimal 6 kata) untuk percakapan berdasarkan respons AI berikut. "
        "Balas hanya judul tanpa tanda kutip:\n\n"
        f"{response_text}"
    )
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            extra_body=build_extra_body(provider_name, False),
        )
        title = completion.choices[0].message.content.strip()
        return title[:80] if title else "New chat"
    except Exception:
        fallback = response_text.strip().split("\n")[0]
        return fallback[:40] + ("..." if len(fallback) > 40 else "") if fallback else "New chat"


def parse_data_file(file_path: Path) -> dict:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        pdf = PdfReader(str(file_path))
        content = "\n".join(page.extract_text() or "" for page in pdf.pages)
        parsed = content
    else:
        content = file_path.read_text(encoding="utf-8")

    if suffix == ".json":
        parsed = json.loads(content)
    elif suffix in {".txt", ".csv"}:
        parsed = content
    elif suffix == ".pdf":
        parsed = content
    else:
        parsed = content

    return {
        "name": file_path.name,
        "type": suffix,
        "raw": content,
        "parsed": parsed,
    }


def append_data_to_chat(chat: dict, data_items: list[dict]) -> None:
    if not data_items:
        return

    for item in data_items:
        chunks = split_with_overlap_8192_100(item["raw"])
        if len(chunks) == 1:
            chat["messages"].append(
                {
                    "role": "user",
                    "content": f"Data dari file {item['name']}:\n{chunks[0]}",
                }
            )
        else:
            for index, chunk in enumerate(chunks, start=1):
                chat["messages"].append(
                    {
                        "role": "user",
                        "content": (
                            f"Data dari file {item['name']} (chunk {index}/{len(chunks)}):\n{chunk}"
                        ),
                    }
                )


# --- Init DB ---
init_db()

# --- Session state init ---
if "provider_name" not in st.session_state:
    st.session_state.provider_name = DEFAULT_PROVIDER
if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL_BY_PROVIDER[st.session_state.provider_name]

if "chats" not in st.session_state:
    loaded_chats, loaded_order = load_chats_from_db()
    if loaded_chats:
        st.session_state.chats = loaded_chats
        st.session_state.chat_order = loaded_order
        st.session_state.active_chat_id = loaded_order[0]
    else:
        first_chat_id = str(uuid.uuid4())
        st.session_state.chats = {first_chat_id: create_chat("New chat")}
        st.session_state.chat_order = [first_chat_id]
        st.session_state.active_chat_id = first_chat_id
        save_chat_to_db(first_chat_id, st.session_state.chats[first_chat_id])

if "loaded_data" not in st.session_state:
    st.session_state.loaded_data = []

current_chat = st.session_state.chats[st.session_state.active_chat_id]

# --- Sidebar ---
with st.sidebar:
    st.subheader("⚙️ Settings")

    # Provider selector
    provider_name = st.selectbox(
        "Provider",
        options=list(PROVIDERS.keys()),
        index=list(PROVIDERS.keys()).index(st.session_state.provider_name)
        if st.session_state.provider_name in PROVIDERS
        else 0,
    )

    # If provider changes, reset model to provider default
    if provider_name != st.session_state.provider_name:
        st.session_state.provider_name = provider_name
        st.session_state.model_name = DEFAULT_MODEL_BY_PROVIDER[provider_name]
        st.rerun()

    # Model selector (depends on provider)
    model_options = PROVIDERS[provider_name]["models"]
    model_name = st.selectbox(
        "Model",
        options=model_options,
        index=model_options.index(st.session_state.model_name)
        if st.session_state.model_name in model_options
        else 0,
    )
    st.session_state.model_name = model_name

    if st.button("+ New chat", use_container_width=True, help="Buat sesi chat baru"):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chats[new_chat_id] = create_chat("New chat")
        st.session_state.chat_order.insert(0, new_chat_id)
        st.session_state.active_chat_id = new_chat_id
        save_chat_to_db(new_chat_id, st.session_state.chats[new_chat_id])
        st.rerun()

    if st.button(
        "📥 Ambil data",
        use_container_width=True,
        help="Muat file .json/.csv/.txt/.pdf dari folder ./data",
    ):
        if not DATA_DIR.exists() or not DATA_DIR.is_dir():
            st.warning("Folder ./data tidak ditemukan.")
        else:
            data_files = [
                p
                for p in sorted(DATA_DIR.iterdir())
                if p.suffix.lower() in {".json", ".csv", ".txt", ".pdf"}
            ]
            processed_items = [parse_data_file(file_path) for file_path in data_files]
            st.session_state.loaded_data = processed_items
            append_data_to_chat(current_chat, processed_items)
            save_chat_to_db(st.session_state.active_chat_id, current_chat)
            st.success(f"{len(processed_items)} file data berhasil diproses ke session state.")
            st.rerun()

    enable_thinking = st.toggle("Enable thinking", value=True)

    st.caption(f"Provider: `{provider_name}`")
    st.caption(f"Model aktif: `{model_name}`")
    st.caption(f"Total chunk session: `{count_total_session_chunks(current_chat['messages'])}`")

    st.divider()
    st.subheader("🗂️ List session chat")
    st.caption(f"Total sesi: {len(st.session_state.chat_order)}")

    for chat_id in st.session_state.chat_order:
        chat = st.session_state.chats[chat_id]
        chat_title = chat["title"] if len(chat["title"]) <= 32 else f"{chat['title'][:32]}..."
        chat_label = f"🟢 {chat_title}" if chat_id == st.session_state.active_chat_id else chat_title
        if st.button(
            chat_label,
            key=f"chat_{chat_id}",
            use_container_width=True,
            type="primary" if chat_id == st.session_state.active_chat_id else "secondary",
            help=f"{len(chat['messages'])} pesan",
        ):
            st.session_state.active_chat_id = chat_id
            st.rerun()

# --- Client follows provider selection ---
client = get_client(st.session_state.provider_name)

if not current_chat["messages"]:
    st.markdown(
        '<div class="chat-tip">Mulai percakapan dengan menulis pertanyaan di bawah. Untuk teks panjang, app akan otomatis memecahnya menjadi beberapa chunk token.</div>',
        unsafe_allow_html=True,
    )

# --- Render history ---
for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input ---
user_text = st.chat_input("Tulis pertanyaan...")

if user_text:
    if not client:
        provider_name = st.session_state.get("provider_name", DEFAULT_PROVIDER)
        needed_key = PROVIDERS[provider_name]["env_key"]
        st.error(f"Chat tidak dapat diproses karena API key `{needed_key}` belum tersedia.")
        st.stop()

    # Tidak ada limit token per user input: simpan apa adanya sebagai satu pesan user.
    current_chat["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    save_chat_to_db(st.session_state.active_chat_id, current_chat)

    # --- Assistant streaming ---
    with st.chat_message("assistant"):
        thinking_expander = st.expander("Thinking", expanded=True)
        thinking_box = thinking_expander.empty()
        answer_box = st.empty()

        thinking_text = ""
        answer_text = ""
        raw_answer_text = ""
        tag_thinking_text = ""

        provider_name = st.session_state.get("provider_name", DEFAULT_PROVIDER)
        model_name = st.session_state.get(
            "model_name", DEFAULT_MODEL_BY_PROVIDER.get(provider_name, "deepseek-chat")
        )

        completion = client.chat.completions.create(
            model=model_name,
            messages=current_chat["messages"],
            extra_body=build_extra_body(provider_name, enable_thinking),
            stream=True,
        )

        for chunk in completion:
            delta = chunk.choices[0].delta

            # More robust: provider may use different naming
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            content = getattr(delta, "content", None)

            if reasoning:
                thinking_text += reasoning
                thinking_box.markdown(thinking_text)

            if content:
                raw_answer_text += content
                answer_text, tag_thinking_text = strip_thinking_blocks(raw_answer_text)
                rendered_thinking = "\n\n".join(
                    [part for part in [thinking_text.strip(), tag_thinking_text.strip()] if part]
                )
                if rendered_thinking:
                    thinking_box.markdown(rendered_thinking)
                answer_box.markdown(answer_text)

        assistant_count_before_append = len(
            [message for message in current_chat["messages"] if message["role"] == "assistant"]
        )
        current_chat["messages"].append({"role": "assistant", "content": answer_text})

        # Title only from first assistant message in this chat
        if assistant_count_before_append == 0:
            current_chat["title"] = generate_title_from_first_ai_response(
                client, provider_name, model_name, answer_text
            )

        save_chat_to_db(st.session_state.active_chat_id, current_chat)
