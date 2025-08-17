import streamlit as st
import time
from src.rag_pipeline import load_pipeline

st.set_page_config(page_title="Amlgo RAG Chatbot", page_icon="🤖", layout="wide")

# Custom CSS for chat bubbles
st.markdown(
    """
    <style>
    .chat-bubble {
        max-width: 75%;
        padding: 10px 15px;
        border-radius: 20px;
        margin: 8px 0;
        line-height: 1.4;
        font-size: 15px;
        word-wrap: break-word;
    }
    .user-bubble {
        background-color: #DCF8C6;
        margin-left: auto;
        text-align: right;
    }
    .assistant-bubble {
        background-color: #F1F0F0;
        margin-right: auto;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cache pipeline load
@st.cache_resource
def get_pipeline():
    return load_pipeline()

main_chain, retriever = get_pipeline()

# Sidebar
with st.sidebar:
    st.title("⚙️ Chatbot Settings")
    st.markdown("**Model in use:** LLaMA-3.1-8B-Instruct")
    st.markdown("**Retriever:** FAISS with MiniLM embeddings")
    if st.button("Clear Chat"):
        st.session_state.messages = []

# Title
st.title("📘 Amlgo Labs RAG Chatbot")

# Initialize session state with greeting
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi, I’m your AI assistant trained on the Amlgo Labs training document. How can I help you today?"}
    ]

# Render past messages
for msg in st.session_state.messages:
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='chat-bubble {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# User input
user_input = st.chat_input("Type your query here...")

if user_input:
    # Save user msg
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"<div class='chat-bubble user-bubble'>{user_input}</div>", unsafe_allow_html=True)

    # Streaming assistant response
    streamed_text = ""
    answer = main_chain.invoke(user_input)

    response_placeholder = st.empty()
    for token in answer.split():
        streamed_text += token + " "
        time.sleep(0.03)
        response_placeholder.markdown(f"<div class='chat-bubble assistant-bubble'>{streamed_text}▌</div>", unsafe_allow_html=True)

    response_placeholder.markdown(f"<div class='chat-bubble assistant-bubble'>{streamed_text}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": streamed_text})

    # Sources
    sources = retriever.get_relevant_documents(user_input)
    if sources:
        with st.expander("📄 Sources"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {src.page_content[:500]}...")
