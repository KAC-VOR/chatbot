
import os
import tempfile

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI

#from streamlit import caching

#거의 최종버전
################### 배포 때문에 추가
###https://discuss.streamlit.io/t/chromadb-sqlite3-your-system-has-an-unsupported-version-of-sqlite3/90975

import pysqlite3
import sys
import sqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

#########################
#오픈AI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
id = st.secrets['id']

def cache_clear():
    #st.cache_data.clear()  # Clears @st.cache_data cache
    st.cache_resource.clear()  # Clears @st.cache_resource cache
    #caching.clear_cache()
#cache_resource로 한번 실행한 결과 캐싱해두기
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

#텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장


@st.cache_resource
def load_pdf(_file):

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
        tmp_file.write(_file.getvalue())
        tmp_file_path = tmp_file.name
        #PDF 파일 업로드
        loader = PyPDFLoader(file_path=tmp_file_path)
        pages = loader.load_and_split()
    return pages

@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(_docs)
    #persist_directory = "./chroma_pdf_db"
    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(model='text-embedding-3-small'),
        #   persist_directory=persist_directory
    )
    return vectorstore
#만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vectorstore():
    persist_directory = "./chroma_db"
    print(persist_directory)
    #if os.path.exists(persist_directory):
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')
    )
    #else:
     #   return 0

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# PDF 문서 로드-벡터 DB 저장-검색기-히스토리 모두 합친 Chain 구축
@st.cache_resource
def initialize_components(selected_model):
    #file_path = r"../data/"
    #file_path = r"C:/Users/Jay/PycharmProjects/test_ai/input3.pdf"
    #pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()

    # 채팅 히스토리 요약 시스템 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 질문-답변 시스템 프롬프트
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    Please say answer only based pieces of retrieved context.\
    If you don't know the answers, Please say you don't know. \
    Keep the answer perfect. please use imogi with the answer.\
    대답은 한국어로 하고, 존댓말을 써줘.\
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
#토큰제한 어떻게하지?
    llm = ChatOpenAI(model=selected_model,temperature=0)
    #ChatOpenAI(model_name='gpt-4o',
    #    n=1,stop=None, temperature=1.0, api_key=OPENAI_API_KEY)
    #temp가 0이면 거짓말을 하지 않고 문서기반으로 답변을 함
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


#retriever = vectorstore.as_retriever(
#    search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.6})
#@st.cache_resource
def initial_not_select(selected_model):
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요 챗봇입니다. 무엇을 도와드릴까요?"}]
    print(st.session_state)
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
   # prompt=""
    if prompt := st.chat_input(key=1):
        client = OpenAI()
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = client.chat.completions.create(model=selected_model, messages=st.session_state.messages)
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

@st.cache_resource
def chaining(_pages,selected_model):
    vectorstore = create_vector_store(_pages)
    retriever = vectorstore.as_retriever()
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}) 동일함
    # 채팅 히스토리 요약 시스템 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
       which might reference context in the chat history, formulate a standalone question \
       which can be understood without the chat history. Do NOT answer the question, \
       just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    #이 부분의 시스템 프롬프트는 기호에 따라 변경하면 됩니다.
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    Please say answer only based pieces of retrieved context.\
    If you don't know the answers, Please say you don't know. \
    Keep the answer perfect. please use imogi with the answer.\
    대답은 한국어로 하고, 존댓말을 써줘.\
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model=selected_model,temperature=0)
    #ChatOpenAI(model_name='gpt-4o',
    #    n=1,stop=None, temperature=1.0, api_key=OPENAI_API_KEY)
    #temp가 0이면 거짓말을 하지 않고 문서기반으로 답변을 함
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain



if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

#login

def login(id_in):
    if id == id_in:
        st.session_state["logged_in"] = True
        st.success("login succes")
        st.rerun()
    else:
        st.error(("Wrong ID and Password"))

if not st.session_state["logged_in"]:
    st.title("Login1")
    id_in = st.text_input("ID")
    pw = st.text_input("password", type="password")
    if st.button("Login"):
        cache_clear()
        print(id_in)
        print(id)
        login(id_in)
else:


        # Streamlit UI
    st.header("항공통신소 Q&A 챗봇 💬")
    selection = st.selectbox("ChatGpt,기존 Database(노하우 등), PDF ", ("ChatGpt", "Database", "PDF"))
    option = st.selectbox("Select GPT Model", ("gpt-4.1-mini", "gpt-4.1"))
    #st.slider('몇살인가요?', 0, 130, 25)
    #halu_t = st.slider("기존 문서로 답변: 0, 창의력 추가 답변: 1", 0.0,1.0,(0.0))
    #halu = st.selectbox("기존 문서로 답변: 0, 창의력 추가 답변: 1",("0","0.5","1"))

    #halu= str(halu_t)

    #print(halu_t)
    if selection =="ChatGpt":
        cache_clear()
        initial_not_select(option)

    if selection == "PDF":
        #cache_clear()
        uploaded_file = st.file_uploader("PDF 기반 답변", type=["pdf"],accept_multiple_files=True)
        for file in uploaded_file:
            pages = load_pdf(file)
            print(pages)
            print(type(pages))
        try:
            rag_chain = chaining(pages, option)
           # print(rag_chain)
            chat_history = StreamlitChatMessageHistory(key="chat_messages")
            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant",
                                                 "content": "무엇이든 물어!"}]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                lambda session_id: chat_history,
                input_messages_key="input",
                history_messages_key="history",
                output_messages_key="answer",
            )


            for msg in chat_history.messages:
                st.chat_message(msg.type).write(msg.content)

            if prompt_message := st.chat_input("Your question"):
                st.chat_message("human").write(prompt_message)
                with st.chat_message("ai"):
                    with st.spinner("Thinking..."):
                        config = {"configurable": {"session_id": "any"}}
                        response = conversational_rag_chain.invoke(
                            {"input": prompt_message},
                            config)

                        answer = response['answer']
                        st.write(answer)
                        #with st.expander("참고 문서 확인"):
                         #   for doc in response['context']:
                          #      st.markdown(doc.metadata['source'], help=doc.page_content)
        except:
            st.header("📚 PDF 업로드 해주세요!")
            #st.header("한번에 여러 파일 업로드")

    elif selection == "Database":
        #cache_clear()
        rag_chain = initialize_components(option)
        chat_history = StreamlitChatMessageHistory(key="chat_messages")


        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )
        print(st.session_state)
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant",
                                             "content": "무엇이든 물어보세요!"}]

        for msg in chat_history.messages:
            st.chat_message(msg.type).write(msg.content)

        if prompt_message := st.chat_input("Your question"):
            st.chat_message("human").write(prompt_message)
            with st.chat_message("ai"):
                with st.spinner("Thinking..."):
                    config = {"configurable": {"session_id": "any"}}
                    response = conversational_rag_chain.invoke(
                        {"input": prompt_message},
                        config)

                    answer = response['answer']
                    st.write(answer)
                    #with st.expander("참고 문서 확인"):
                     #   for doc in response['context']:
                      #      st.markdown(doc.metadata['source'], help=doc.page_content)"""

