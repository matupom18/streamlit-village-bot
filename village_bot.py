from langchain.schema import HumanMessage, SystemMessage
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI
import mysql.connector
import streamlit as st
import os
# โหลด Google API Key จาก secrets.toml
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- Database connection ---
ssl_path = os.path.join(os.path.dirname(__file__), "isrgrootx1.pem")
conn = mysql.connector.connect(
    host=st.secrets["DB_HOST"],
    port=st.secrets["DB_PORT"],
    user=st.secrets["DB_USER"],
    password=st.secrets["DB_PASSWORD"],
    database=st.secrets["DB_NAME"],
    ssl_ca=ssl_path,
    ssl_verify_cert=True,
    ssl_verify_identity=True
)
cursor = conn.cursor()

# --- Load and process documents ---
from pathlib import Path

def load_txt_documents(directory):
    docs = []
    for file_path in Path(directory).rglob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs.extend(loader.load())
    return docs

docs = load_txt_documents("village_docs")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# --- Embedding model ---
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# --- Vector database ---

# สร้าง Chroma แบบ in-memory ชัดเจน
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    client_settings=Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=None,   # 👈 ห้ามใช้ persist!
        anonymized_telemetry=False
    )
)


retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# --- LLM setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    max_output_tokens=512
)

# --- Prompt setup ---
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant for a housing community."
        "Answer residents' questions using information from community documents."
        "Be polite and respond in Thai."
        "You are a virtual assistant in the community."
        "You are friendly, respectful, and helpful."
        "Speak with a warm and courteous tone."
        "Always provide accurate and timely information."
        "Refer to yourself as 'ดิฉัน' and use 'ค่ะ/คะ' appropriately.  "
        "**Do not start every message with greetings such as 'สวัสดีค่ะ' unless the user greets first.**"
    ),
    HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {question}"
    ),
])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": chat_prompt},
)

# --- Public functions for Streamlit app ---

def ask_village_bot(question: str) -> str:
    """Run a query through the QA chain."""
    return qa_chain.run(question)


def is_problem_statement_with_gemini(q: str) -> bool:
    """
    ฟังก์ชันนี้จะใช้โมเดล Gemini ในการวิเคราะห์ว่า
    ข้อความที่ผู้ใช้ถามนั้นเกี่ยวข้องกับปัญหาที่เกิดขึ้นในหมู่บ้านจัดสรรหรือไม่
    รูปแบบของข้อมูลที่ควรบันทึก:
        - ประเภทของปัญหา (เช่น น้ำไม่ไหล, ไฟดับ, ถนนพัง ฯลฯ)
        - สถานที่ที่เกิดปัญหา (ถ้ามี)
        - เวลาที่แจ้งปัญหา (ถ้ามี)
        - ผู้แจ้ง (หากสามารถระบุได้จากบริบท)
        
        ตัวอย่างที่ควรเก็บในความจำ:
        - “ไฟดับที่ซอย 3 ตั้งแต่เมื่อคืน”
        - “ถนนหน้าวัดเป็นหลุมเป็นบ่อ”
        - “น้ำประปาไม่ไหลตั้งแต่เช้าแถวตลาดนัด”
    """
    prompt = f"""
                วิเคราะห์ข้อความต่อไปนี้: '{q}'
                
                ข้อความนี้เกี่ยวข้องกับปัญหาที่เกิดขึ้นในหมู่บ้านจัดสรรหรือไม่? 
                (ตอบเพียงว่า 'ใช่' หรือ 'ไม่')
                
                เกณฑ์การพิจารณา:
                - หากเป็นข้อความที่กล่าวถึงปัญหา เช่น น้ำไม่ไหล ไฟดับ ถนนเสีย น้ำท่วม ฯลฯ ให้ตอบว่า 'ใช่'
                - หากไม่เกี่ยวข้องกับปัญหาในหมู่บ้าน ให้ตอบว่า 'ไม่'
                """
    response = llm.predict(prompt)
    return "ใช่" in response or "yes" in response.lower()


def store_issue_in_db(question: str, answer: str):
    """บันทึกข้อความแจ้งปัญหาไว้ในฐานข้อมูล"""
    query = "INSERT INTO issues (question, answer) VALUES (%s, %s)"
    values = (question, answer)
    cursor.execute(query, values)
    conn.commit()
