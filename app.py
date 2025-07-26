import os
import uuid
from datetime import datetime

import streamlit as st
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as extract_text_pdfminer

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="Bilingual PDF Chatbot", layout="wide")
st.title("📚 Bilingual PDF Chatbot (Bangla + English)")
st.markdown("Upload a Bangla or English PDF and ask questions.")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        except Exception:
            text += extract_text_pdfminer(pdf)
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks, model_name, google_api_key=None, openai_api_key=None, hf_api_key=None, index_path="faiss_index"):
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    elif model_name == "OpenAI":
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    elif model_name == "HuggingFace (Multilingual RAG)":
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    else:
        raise ValueError("Unsupported model.")
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(index_path)
    return vector_store, embeddings


def get_conversational_chain(model_name, google_api_key=None, openai_api_key=None, hf_api_key=None):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, just say, "Answer is not available in the context." Don't make up information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    if model_name == "Google AI":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=google_api_key)
    elif model_name == "OpenAI":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_api_key)
    elif model_name == "HuggingFace (Multilingual RAG)":
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            task="text2text-generation",
            model_kwargs={"temperature": 0.1, "max_length": 512},
            huggingfacehub_api_token=hf_api_key
        )
    else:
        raise ValueError("Unsupported model.")

    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)


def user_input(user_question, model_name, google_api_key, openai_api_key, hf_api_key, pdf_docs, conversation_history):
    if not pdf_docs or \
       (model_name == "Google AI" and not google_api_key) or \
       (model_name == "OpenAI" and not openai_api_key) or \
       (model_name == "HuggingFace (Multilingual RAG)" and not hf_api_key):
        st.warning("Please upload PDFs and provide a valid API key.")
        return

    if "vector_store" not in st.session_state:
        text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(text)
        index_path = f"faiss_index_{uuid.uuid4()}"
        vector_store, embeddings = get_vector_store(text_chunks, model_name, google_api_key, openai_api_key, hf_api_key, index_path=index_path)
        st.session_state.vector_store = vector_store
        st.session_state.embeddings = embeddings
        st.session_state.index_path = index_path
    else:
        vector_store = st.session_state.vector_store
        embeddings = st.session_state.embeddings

    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain(model_name, google_api_key, openai_api_key, hf_api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    response_output = response["output_text"]
    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []

    conversation_history.append((user_question, response_output, model_name,
                                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                 ", ".join(pdf_names)))

    st.markdown(f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{ background-color: #2b313e; }}
            .chat-message.bot {{ background-color: #475063; }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
            </div>
            <div class="message">{response_output}</div>
        </div>
    """, unsafe_allow_html=True)


#SIDEBAR
st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### 🔐 API Keys")
api_key_google = st.sidebar.text_input("Google API Key", type="password", value=GOOGLE_API_KEY or "")
api_key_openai = st.sidebar.text_input("OpenAI API Key", type="password", value=OPENAI_API_KEY or "")
hf_api_key = st.sidebar.text_input("HuggingFace API Token", type="password", value=HUGGINGFACEHUB_API_TOKEN or "")

st.sidebar.markdown("### 🧠 Model & Files")
model_choice = st.sidebar.selectbox("Select Model", ["Google AI", "OpenAI", "HuggingFace (Multilingual RAG)"])
uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.conversation_history = []
    for key in ["vector_store", "embeddings", "index_path"]:
        if key in st.session_state:
            del st.session_state[key]

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

#Ask UI
user_question = st.text_input("📝 Type your question and press Enter:")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    ask_button = st.button("💬 Ask", use_container_width=True)

if ask_button and user_question:
    user_input(user_question, model_choice, api_key_google, api_key_openai, hf_api_key, uploaded_files, st.session_state.conversation_history)

#HISTORY
if st.session_state.conversation_history:
    with st.expander("🕘 Conversation History", expanded=False):
        for q, a, model, time, pdfs in st.session_state.conversation_history:
            st.markdown(f"**[{time}]** Model: `{model}` | PDFs: `{pdfs}`")
            st.markdown(f"**Q:** {q}\n\n**A:** {a}")
            st.markdown("---")
