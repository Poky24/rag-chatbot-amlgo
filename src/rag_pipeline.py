import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def load_pipeline(pdf_name="AI Training Document.pdf"):
    # Paths
    root_dir = os.path.dirname(os.path.dirname(__file__))
    pdf_path = os.path.join(root_dir, "data", pdf_name)
    chunks_dir = os.path.join(root_dir, "chunks")
    vectordb_dir = os.path.join(root_dir, "vectordb")

    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(vectordb_dir, exist_ok=True)

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = text_splitter.create_documents([doc.page_content for doc in docs])

    # Save chunks
    chunks_path = os.path.join(chunks_dir, "chunks.json")
    if not os.path.exists(chunks_path):
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump([c.page_content for c in chunks], f, ensure_ascii=False, indent=2)

    # Embeddings + FAISS (persisting)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb_path = os.path.join(vectordb_dir, "faiss_index")

    if os.path.exists(vectordb_path):
        vector_store = FAISS.load_local(vectordb_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(vectordb_path)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # LLM
    llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
    model = ChatHuggingFace(llm=llm)

    # Prompt
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

    # RAG Chain
    parallel_chain = RunnableParallel(
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    )
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | model | parser

    return main_chain, retriever
