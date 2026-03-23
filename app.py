import streamlit as st
import os
import re
import sqlite3
import uuid
from datetime import datetime, UTC
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multi-Provider AI Chat", page_icon="🤖", layout="wide")
st.title("💬 Multi-Provider AI Chat")
st.caption("A chat application that automatically routes processes based on the selected model with SQLite history.")

DB_PATH = "chat_history.db"

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
    chats = {}
    chat_order = []

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

        for seq, message in enumerate(chat["messages"], start=1):
            conn.execute(
                """
                INSERT INTO messages (chat_id, role, content, seq, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, message["role"], message["content"], seq, now),
            )

        conn.commit()

def create_new_chat() -> None:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"title": "New chat", "messages": []}
    st.session_state.chat_order.insert(0, chat_id)
    st.session_state.active_chat_id = chat_id
    st.session_state.total_tokens = 0
    save_chat_to_db(chat_id, st.session_state.chats[chat_id])

def initialize_session_state() -> None:
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    if "chats" not in st.session_state:
        chats, order = load_chats_from_db()
        if chats:
            st.session_state.chats = chats
            st.session_state.chat_order = order
            st.session_state.active_chat_id = order[0]
        else:
            st.session_state.chats = {}
            st.session_state.chat_order = []
            create_new_chat()

def generate_title_from_first_ai_response(api_key: str, base_url: str, model_name: str, response_text: str) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
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

def stream_deepseek_official(api_key: str, messages: list, model_name: str, reasoning_box, enable_reasoning: bool, token_placeholder):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    extra_body = {}
    if enable_reasoning:
        extra_body = {
            "thinking": {
                "type": "enabled"
            }
        }
    
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            extra_body=extra_body if extra_body else None,
            stream_options={"include_usage": True}
        )
        
        thinking_text = ""
        for chunk in stream:
            if getattr(chunk, 'usage', None) is not None:
                st.session_state.total_tokens = chunk.usage.total_tokens
                token_placeholder.metric("Total Tokens (Context)", f"{st.session_state.total_tokens:,}")

            if not getattr(chunk, 'choices', None) or len(chunk.choices) == 0: continue
            delta = chunk.choices[0].delta
            
            reasoning = getattr(delta, 'reasoning_content', None)
            if reasoning and enable_reasoning:
                thinking_text += reasoning
                reasoning_box.info(f"**💭 Thinking Process (DeepSeek):**\n\n{thinking_text}")
                
            content = getattr(delta, 'content', None)
            if content:
                yield content
                
    except Exception as e:
        st.error(f"DeepSeek API Error: {e}")
        yield ""

def stream_openrouter(api_key: str, messages: list, model_name: str, reasoning_box, enable_reasoning: bool, token_placeholder):
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    
    extra_body = {}
    if enable_reasoning:
        extra_body = {
            "reasoning": {
                "enabled": True,
                "exclude": False,
            }
        }
    
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            extra_body=extra_body if extra_body else None,
            stream_options={"include_usage": True}
        )
        
        thinking_text = ""
        raw_content = ""
        
        for chunk in stream:
            # 1. Menangani object usage yang dikirim pada chunk terakhir
            if getattr(chunk, 'usage', None) is not None:
                if isinstance(chunk.usage, dict):
                    st.session_state.total_tokens = chunk.usage.get("total_tokens", 0)
                else:
                    st.session_state.total_tokens = getattr(chunk.usage, "total_tokens", 0)
                
                token_placeholder.metric("Total Tokens (Context)", f"{st.session_state.total_tokens:,}")

            # 2. Skip jika choices kosong untuk menghindari IndexError pada chunk terakhir
            if not getattr(chunk, 'choices', None) or len(chunk.choices) == 0: 
                continue
                
            delta = chunk.choices[0].delta
            
            # 3. Tangkap Reasoning
            reasoning = getattr(delta, 'reasoning', None)
            if reasoning and enable_reasoning:
                thinking_text += reasoning
                reasoning_box.info(f"**💭 Thinking Process (OpenRouter):**\n\n{thinking_text}")
            
            # 4. Tangkap Content
            content = getattr(delta, 'content', None)
            if content:
                raw_content += content
                
                clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL | re.IGNORECASE)
                
                if enable_reasoning and len(clean_content) < len(raw_content):
                    think_blocks = re.findall(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL | re.IGNORECASE)
                    if think_blocks:
                        leaked_thinking = "\n".join(think_blocks)
                        reasoning_box.info(f"**💭 Thinking Process (<think> tags):**\n\n{leaked_thinking}")
                
                if not raw_content.rfind("<think>") > raw_content.rfind("</think>"):
                    yield content.replace("<think>", "").replace("</think>", "")
                    
    except Exception as e:
        st.error(f"OpenRouter API Error: {e}")
        yield ""

init_db()
initialize_session_state()

current_chat = st.session_state.chats[st.session_state.active_chat_id]

with st.sidebar:
    st.header("⚙️ Chat Settings")
    
    model_options = [
        "deepseek-chat",
        "qwen/qwen3.5-plus-02-15",
        "qwen/qwen3.5-flash-02-23",
        "google/gemini-3-flash-preview"
    ]
    selected_model = st.selectbox("Select Model:", model_options)
    
    if selected_model == "deepseek-chat":
        provider = "Official DeepSeek"
        base_url = "https://api.deepseek.com"
        api_key = st.text_input("DeepSeek API Key", type="password", value=os.getenv("DEEPSEEK_API_KEY", ""))
    else:
        provider = "OpenRouter"
        base_url = "https://openrouter.ai/api/v1"
        api_key = st.text_input("OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
    
    st.caption(f"*Active Provider: {provider}*")
    st.divider()
    
    enable_reasoning_radio = st.radio("Thinking Process (Reasoning):", ["Off", "On"], index=0)
    enable_reasoning = True if enable_reasoning_radio == "On" else False

    st.divider()
    token_placeholder = st.empty()
    token_placeholder.metric("Total Tokens (Context)", f"{st.session_state.total_tokens:,}")

    st.divider()
    st.subheader("🗂️ Chat Sessions")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()
    
    st.divider()
    
    for chat_id in st.session_state.chat_order:
        chat_data = st.session_state.chats[chat_id]
        is_active = chat_id == st.session_state.active_chat_id
        title = chat_data["title"] if len(chat_data["title"]) <= 32 else f"{chat_data['title'][:32]}..."
        label = f"🟢 {title}" if is_active else title

        if st.button(
            label,
            key=f"chat_{chat_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            help=f"{len(chat_data['messages'])} messages",
        ):
            st.session_state.active_chat_id = chat_id
            st.session_state.total_tokens = 0
            st.rerun()

if not api_key:
    st.info(f"💡 Please enter your API Key for {provider} in the sidebar to start.")
    st.stop()

if not current_chat["messages"]:
    st.info("Start the conversation by typing a question below.")

for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            display_text = msg["content"]
            if "\n--- Document Content:" in display_text:
                user_msg_part = display_text.split("\n--- Document Content:")[0].strip()
                if user_msg_part:
                    st.markdown(user_msg_part)
                st.caption("📎 *(Document attached in AI context)*")
            else:
                st.markdown(display_text)
        else:
            st.markdown(msg["content"])

prompt_data = st.chat_input(
    "Type your message and/or upload files...",
    accept_file="multiple",
    file_type=["pdf", "json", "txt", "csv", "tsv"]
)

if prompt_data:
    user_text = prompt_data.text if prompt_data.text else ""
    uploaded_files = prompt_data.files if hasattr(prompt_data, 'files') and prompt_data.files else []
    
    final_prompt = user_text
    file_contents = []
    
    if uploaded_files:
        for file in uploaded_files:
            try:
                ext = file.name.split('.')[-1].lower()
                if ext == "pdf":
                    reader = PyPDF2.PdfReader(file)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                else:
                    text = file.read().decode("utf-8")
                    
                file_contents.append(f"\n--- Document Content: {file.name} ---\n{text}")
            except Exception as e:
                st.error(f"Failed to read file {file.name}: {e}")
                
        if file_contents:
            final_prompt += "".join(file_contents)
            
    if final_prompt.strip():
        with st.chat_message("user"):
            if user_text:
                st.markdown(user_text)
            for file in uploaded_files:
                st.caption(f"📎 File uploaded: {file.name}")
        
        current_chat["messages"].append({"role": "user", "content": final_prompt})
        save_chat_to_db(st.session_state.active_chat_id, current_chat)
        
        with st.chat_message("assistant"):
            reasoning_box = st.empty()
            content_box = st.empty()
            
            with content_box:
                if selected_model == "deepseek-chat":
                    generator = stream_deepseek_official(api_key, current_chat["messages"], selected_model, reasoning_box, enable_reasoning, token_placeholder)
                else:
                    generator = stream_openrouter(api_key, current_chat["messages"], selected_model, reasoning_box, enable_reasoning, token_placeholder)
                
                full_response = st.write_stream(generator)
            
            current_chat["messages"].append({"role": "assistant", "content": full_response})
            
            assistant_count = sum(1 for m in current_chat["messages"] if m["role"] == "assistant")
            if assistant_count == 1 and current_chat.get("title", "New chat") == "New chat" and full_response.strip():
                new_title = generate_title_from_first_ai_response(api_key, base_url, selected_model, full_response)
                current_chat["title"] = new_title

            save_chat_to_db(st.session_state.active_chat_id, current_chat)
            st.rerun()