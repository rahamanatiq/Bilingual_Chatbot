from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from pydantic import BaseModel
import uuid

from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as extract_text_pdfminer

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

app = FastAPI()

#HELPERS
def get_pdf_text(files: List[UploadFile]) -> str:
    text = ""
    for pdf in files:
        try:
            reader = PdfReader(pdf.file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except:
            text += extract_text_pdfminer(pdf.file)
    return text

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)

def get_llm_chain():
    prompt_template = """
    Answer the question based on the following context. If not found, say "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        task="text2text-generation",
        model_kwargs={"temperature": 0.1, "max_length": 512},
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

#API ENDPOINT
@app.post("/ask/")
async def ask_question(question: str = Form(...), files: List[UploadFile] = File(...)):
    text = get_pdf_text(files)
    chunks = get_chunks(text)
    vector_store = build_vector_store(chunks)
    docs = vector_store.similarity_search(question, k=3)

    # Evaluation Scores
    context_text = "\n".join([doc.page_content for doc in docs])
    similarity_score = vector_store.similarity_search_with_score(question, k=1)[0][1]

    chain = get_llm_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return {
        "question": question,
        "response": response["output_text"],
        "context": context_text,
        "similarity_score": similarity_score
    }
