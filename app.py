import streamlit as st
from datetime import datetime
from langchain.schema import HumanMessage, SystemMessage
from village_bot import ask_village_bot, is_problem_statement_with_gemini, store_issue_in_db

st.set_page_config(page_title="Village Chatbot", page_icon="🏡", layout="centered")

st.title("🤖 ผู้ช่วยเสมือนหมู่บ้านจัดสรร")
st.markdown("พูดคุยหรือสอบถามเกี่ยวกับปัญหาในหมู่บ้านได้ที่นี่")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display past messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input prompt
if user_input := st.chat_input("พิมพ์ข้อความของคุณที่นี่..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("กำลังตอบกลับ..."):
            response = ask_village_bot(user_input)
            st.markdown(response)

    # Save to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Check and log issue if relevant
    if is_problem_statement_with_gemini(user_input):
        store_issue_in_db(user_input, response)
