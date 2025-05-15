import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import pandas as pd
import tempfile
import os

# Load .env file
load_dotenv()

st.set_page_config(page_title="Simple RAG App", page_icon="🔍")

st.title("📚 Simple RAG: Upload File and Ask Questions")

# Sidebar - File uploader
uploaded_file = st.sidebar.file_uploader("Upload your PDF or CSV", type=["pdf", "csv"])

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.retriever = None

vectorstore_path = "vectorstore_db"

# Check if vectorstore exists and load if available
if os.path.exists(vectorstore_path) and not uploaded_file:
    embeddings = OpenAIEmbeddings()
    st.session_state.vector_store = FAISS.load_local(vectorstore_path, embeddings,
                                                     allow_dangerous_deserialization=True)
    st.session_state.retriever = st.session_state.vector_store.as_retriever()
    st.success("✅ Loaded existing vector store!")

# Process new file upload
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    # Load documents
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(tmp_path)
        content = df.to_csv(index=False)
        documents = [Document(page_content=content)]
    else:
        st.error("Unsupported file type!")
        st.stop()

    # Split documents
    splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    docs = splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)

    # Save locally
    vector_store.save_local(vectorstore_path)

    st.session_state.vector_store = vector_store
    st.session_state.retriever = vector_store.as_retriever()

    st.success("✅ File uploaded, indexed, and saved locally!")

# Chat Interface
if st.session_state.retriever:
    user_question = st.text_input("Ask a question about your file:")

    if user_question:
        llm = ChatOpenAI(model_name="gpt-4-turbo-2024-04-09", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.retriever)
        answer = qa_chain.run(user_question)

        st.markdown(f"**Answer:** {answer}")
else:
    st.info("👈 Please upload a file first or load an existing vector store.")
