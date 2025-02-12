import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure HF_TOKEN is loaded properly
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("HF_TOKEN is missing. Ensure it's set in your .env file.")
    st.stop()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("Conversation with RAG - PDF Uploads & Chat History")
st.write("Upload PDFs and chat with their content.")

# Input Groq API Key
api_key = st.text_input("Enter your Groq API Key: ", type="password")

if api_key:
    try:
        llm = ChatGroq(groq_api_key=api_key, model="deepseek-r1-distill-llama-70b")
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {str(e)}")
        st.stop()

    # Chat interface
    session_id = st.text_input("Session ID", value="default_session")

    # Manage session chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.getvalue())
                temp_pdf_path = temp_pdf.name  # Get the path of the temporary file

            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()

        # Contextualize system prompt
        contextualize_q_system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Rephrase the user question to be standalone without chat history."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_system_prompt)

        # ✅ Fixed: Ensure `context` is included
        system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Use the retrieved context to answer concisely. If unknown, say you don't know.\n\nRetrieved Context:\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create QA Chain
        question_answer_chain = create_stuff_documents_chain(llm, system_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Session History Management
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        # Create RAG Chain with History
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User Input for Q&A
        user_input = st.text_input("Your question:")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            st.write("### Chat History:")
            for msg in session_history.messages:
                st.write(msg)  # Display stored messages

            st.success(f"Assistant: {response['answer']}")  # ✅ Corrected response format

else:
    st.warning("Please enter the Groq API Key.")
