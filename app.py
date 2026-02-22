import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
import os


def get_pdf_text(uploaded_files):
    raw_text = ""

    for file in uploaded_files:
        pdf_reader = PdfReader(file)

        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

    return raw_text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    return text_splitter.split_text(raw_text)


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    return FAISS.from_texts(text_chunks, embeddings)


def get_conversation_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Answer:
    """)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_question(x):
        return x["question"]

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": get_question,
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm
    )

    return chain


def handle_user_input(user_question):

    inputs = {
        "question": user_question,
        "chat_history": st.session_state.chat_history
    }

    with st.spinner("Thinking..."):
        response = st.session_state.conversation_chain.invoke(inputs)
        answer = response.content

    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("assistant", answer))


def main():

    load_dotenv()

    st.set_page_config(
        page_title="Multi-PDF AI Assistant",
        page_icon="📚",
        layout="wide"
    )

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("📚 Multi-PDF AI Assistant")
    st.markdown("Upload PDFs and ask questions about them.")

    col1, col2 = st.columns([3, 1])

    with col1:

        st.subheader("💬 Chat")

        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        user_question = st.chat_input("Ask something about your documents...")

        if user_question:

            if st.session_state.conversation_chain is None:
                st.error("Process PDFs first before asking questions.")
            else:
                handle_user_input(user_question)
                st.rerun()

    with st.sidebar:

        st.header("📂 Document Manager")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded")
        else:
            st.warning("No files uploaded")

        st.divider()

        if st.button("🚀 Process Documents", use_container_width=True):

            if not uploaded_files:
                st.error("Upload PDFs first!")
            else:
                with st.spinner("Processing documents..."):

                    raw_text = get_pdf_text(uploaded_files)
                    chunks = get_text_chunks(raw_text)

                    if len(chunks) == 0:
                        st.error("No text found inside PDFs.")
                    else:
                        vector_store = get_vector_store(chunks)
                        st.session_state.conversation_chain = get_conversation_chain(vector_store)
                        st.session_state.chat_history = []

                st.success("Documents processed successfully!")


if __name__ == "__main__":
    main()