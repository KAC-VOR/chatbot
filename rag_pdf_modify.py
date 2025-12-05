import os
import sys
import tempfile

import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from openai import OpenAI

# ============ 설정 ============
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
VALID_ID = st.secrets["id"]
EMBEDDING_MODEL = "text-embedding-3-small"
PERSIST_DIR = "./chroma_db"

CONTEXTUALIZE_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

QA_PROMPT = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
Please say answer only based pieces of retrieved context.\
If you don't know the answers, Please say you don't know. \
Keep the answer perfect. please use imogi with the answer.\
대답은 한국어로 하고, 존댓말을 써줘.

Context: {context}

Question: {input}"""


# ============ 유틸리티 함수 ============
def clear_cache():
    st.cache_resource.clear()


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever, model: str):
    llm = ChatOpenAI(model=model, temperature=0)
    
    # 히스토리 기반 질문 재구성
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    
    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    # QA 체인
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    
    def get_context(input_dict):
        if input_dict.get("history"):
            question = contextualize_chain.invoke(input_dict)
        else:
            question = input_dict["input"]
        docs = retriever.invoke(question)
        return format_docs(docs)
    
    rag_chain = (
        RunnablePassthrough.assign(context=get_context)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


# ============ 벡터스토어 관련 ============
@st.cache_resource
def load_pdf_pages(file_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        loader = PyPDFLoader(tmp.name)
        return loader.load_and_split()


@st.cache_resource
def create_vectorstore_from_docs(_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(_docs)
    return Chroma.from_documents(split_docs, get_embeddings())


@st.cache_resource
def get_persistent_vectorstore():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())


# ============ 체인 초기화 ============
@st.cache_resource
def initialize_db_chain(model: str):
    vectorstore = get_persistent_vectorstore()
    return create_rag_chain(vectorstore.as_retriever(), model)


@st.cache_resource
def initialize_pdf_chain(_pages, model: str):
    vectorstore = create_vectorstore_from_docs(_pages)
    return create_rag_chain(vectorstore.as_retriever(), model)


# ============ 채팅 UI ============
def render_chat_history(history):
    for msg in history.messages:
        st.chat_message(msg.type).write(msg.content)


def handle_rag_chat(rag_chain, chat_history):
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda _: chat_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    render_chat_history(chat_history)
    
    if prompt := st.chat_input("Your question"):
        st.chat_message("human").write(prompt)
        with st.chat_message("ai"), st.spinner("Thinking..."):
            response = conversational_chain.invoke(
                {"input": prompt},
                {"configurable": {"session_id": "any"}}
            )
            st.write(response)


def handle_chatgpt_mode(model: str):
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요 챗봇입니다. 무엇을 도와드릴까요?"}
        ]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            response = OpenAI().chat.completions.create(
                model=model,
                messages=st.session_state.messages
            )
            answer = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)


# ============ 인증 ============
def login(input_id: str):
    if input_id == VALID_ID:
        st.session_state.logged_in = True
        st.success("로그인 성공!")
        st.rerun()
    else:
        st.error("잘못된 ID입니다.")


def render_login_page():
    st.title("Login")
    input_id = st.text_input("ID")
    st.text_input("Password", type="password")
    if st.button("Login"):
        clear_cache()
        login(input_id)


# ============ 메인 앱 ============
def render_main_app():
    st.header("항공통신소 Q&A 챗봇 💬")
    
    mode = st.selectbox("모드 선택", ("ChatGpt", "Database", "PDF"))
    model = st.selectbox("GPT 모델", ("gpt-4.1-mini", "gpt-4.1"))
    chat_history = StreamlitChatMessageHistory(key="chat_messages")
    
    if mode == "ChatGpt":
        clear_cache()
        handle_chatgpt_mode(model)
    
    elif mode == "PDF":
        uploaded_files = st.file_uploader(
            "PDF 기반 답변", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if not uploaded_files:
            st.info("📚 PDF를 업로드해주세요!")
            return
        
        all_pages = []
        for f in uploaded_files:
            pages = load_pdf_pages(f.getvalue(), f.name)
            all_pages.extend(pages)
        
        if all_pages:
            rag_chain = initialize_pdf_chain(all_pages, model)
            handle_rag_chat(rag_chain, chat_history)
    
    elif mode == "Database":
        rag_chain = initialize_db_chain(model)
        handle_rag_chat(rag_chain, chat_history)


# ============ 엔트리포인트 ============
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if st.session_state.logged_in:
        render_main_app()
    else:
        render_login_page()


if __name__ == "__main__":
    main()
