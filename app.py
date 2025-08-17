import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import time

# ------------------ SETUP ------------------ #
load_dotenv()
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

@st.cache_resource
def load_pipeline():
    # Load PDF
    loader = PyPDFLoader("AI Training Document.pdf")
    docs = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.create_documents([doc.page_content for doc in docs])

    # Embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # LLM
    llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
    model = ChatHuggingFace(llm=llm)

    # Prompt (no repeated intro here)
    prompt = PromptTemplate(
        template="""
        Answer ONLY using the provided pdf context.
        If the context is insufficient, tell the user politely that this is not covered in the training document.
        Always behave professionally.

        {context}
        Question: {question}
        """,
        input_variables=['context','question']
    )

    def format_docs(retrieved_docs):
        return "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Build chain
    parallel_chain = RunnableParallel(
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    )
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | model | parser

    return main_chain, retriever

main_chain, retriever = load_pipeline()

# ------------------ SIDEBAR ------------------ #
with st.sidebar:
    st.title("⚙️ Chatbot Settings")
    st.markdown("**Model in use:** LLaMA-3.1-8B-Instruct")
    st.markdown("**Retriever:** FAISS with MiniLM embeddings")
    if st.button("Clear Chat"):
        st.session_state.messages = []

# ------------------ MAIN CHAT ------------------ #
st.title("📘 Amlgo Labs RAG Chatbot")

# Initialize session state with first bot message
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
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"<div class='chat-bubble user-bubble'>{user_input}</div>", unsafe_allow_html=True)

    # Streaming assistant response
    streamed_text = ""
    answer = main_chain.invoke(user_input)

    response_placeholder = st.empty()
    for token in answer.split():
        streamed_text += token + " "
        time.sleep(0.03)  # simulate streaming
        response_placeholder.markdown(f"<div class='chat-bubble assistant-bubble'>{streamed_text}▌</div>", unsafe_allow_html=True)

    response_placeholder.markdown(f"<div class='chat-bubble assistant-bubble'>{streamed_text}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": streamed_text})

    # Sources
    sources = retriever.get_relevant_documents(user_input)
    if sources:
        with st.expander("📄 Sources"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {src.page_content[:500]}...")
