# -*- coding: utf-8 -*-
# 이 파일은 Streamlit 기반 RAG(검색 증강 생성) 앱의 핵심 코드입니다.
# 주의: 실제 운영 시엔 API 키 같은 민감정보를 절대로 하드코딩하지 말고, 본 코드처럼 st.secrets 등을 사용하세요.

import os  # 운영체제 기능(환경변수, 경로 등)을 사용하기 위한 표준 라이브러리
import tempfile  # 임시 파일을 만들고 관리하기 위한 표준 라이브러리

import streamlit as st  # 간단히 웹앱을 만들 수 있는 프레임워크(사이드바, 입력창, 출력 등 UI 제공)
from langchain.document_loaders import PyPDFLoader  # PDF 파일을 문서 객체들로 불러오는 도구
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 긴 텍스트를 적당한 크기 덩어리(청크)로 자르는 도구
from langchain_core.runnables import RunnablePassthrough  # 체인 실행 시 입력을 그대로 다음 단계에 전달하는 유틸 (여기선 직접 사용하진 않음)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI 임베딩/챗 모델을 LangChain에서 쓰기 위한 래퍼
from langchain_community.vectorstores import Chroma  # 텍스트 임베딩을 저장/검색하는 벡터DB(Chroma) 연동 래퍼
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # LLM에 보낼 프롬프트(질문 형식) 템플릿 구성 도구
from langchain.chains.combine_documents import create_stuff_documents_chain  # 검색된 문서를 답변 프롬프트에 "그냥 끼워 넣는" 체인 생성 함수
from langchain.chains import create_history_aware_retriever, create_retrieval_chain  # 대화이력 인지 검색기/최종 RAG 체인 생성 함수
from langchain_core.runnables.history import RunnableWithMessageHistory  # 체인에 대화 이력(메시지 히스토리)을 붙여 상태 유지하게 하는 도구
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory  # Streamlit 세션에 대화 이력을 저장/로드하는 구현체
from langchain_core.output_parsers import StrOutputParser  # 모델 응답을 문자열로 파싱하는 유틸 (여기선 직접 사용하진 않음)
from openai import OpenAI  # OpenAI 공식 SDK(직접 Chat Completions API를 호출할 때 사용)

# from streamlit import caching  # (주석 처리) Streamlit의 구 캐시 API; 현재는 st.cache_* 권장

# 거의 최종버전이라는 메모. 실제론 지속적으로 개선될 수 있습니다.
################### 배포 때문에 추가 ###################
### 아래 이슈는 Streamlit Cloud/일부 환경에서 sqlite3 버전 불일치로 Chroma가 오류날 때의 우회책입니다.
### 참고 토론: https://discuss.streamlit.io/t/chromadb-sqlite3-your-system-has-an-unsupported-version-of-sqlite3/90975

import pysqlite3  # 파이썬용 SQLite 대체 모듈(내장 sqlite3 대신 사용)
import sys  # 파이썬 인터프리터 및 모듈 시스템 접근용 표준 라이브러리
import sqlite3  # 내장 sqlite3 모듈(아래에서 대체됨)
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # 파이썬이 sqlite3를 불러오려 할 때 pysqlite3를 대신 쓰게끔 바꿉니다.

#########################
# 오픈AI API 키 설정 (절대 코드에 직접 키를 쓰지 말고, Streamlit Secrets를 쓰세요!)
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']  # OpenAI 라이브러리가 참조할 환경변수에 키를 주입
id = st.secrets['id']  # 매우 단순한 로그인 검사용 비밀값(실서비스에선 OAuth 등 안전한 방식 권장)


def cache_clear():  # Streamlit의 리소스 캐시를 지우는 함수 (변경사항 반영/세션 초기화가 필요할 때 사용)
    # st.cache_data.clear()  # 데이터 캐시를 비우는 코드(현재는 사용 안 함)
    st.cache_resource.clear()  # 리소스 캐시(모델, DB 연결 등)를 비웁니다.
    # caching.clear_cache()  # 구버전 캐시 초기화 코드(현재 주석)


# 아래 데코레이터는 함수 결과를 세션 동안 재사용(캐싱)해서 속도 향상을 돕습니다.
@st.cache_resource
def load_and_split_pdf(file_path):  # 파일 경로로 PDF를 로드해 페이지 단위 문서들로 분리하는 함수
    loader = PyPDFLoader(file_path)  # PDF 로더 준비
    return loader.load_and_split()  # PDF를 로드하고 페이지별 문서 리스트로 반환


# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장하는 전 단계: PDF 업로드 처리
@st.cache_resource  # 업로드/파싱 결과를 캐시해 같은 파일로 재실행 시 속도를 높입니다.
def load_pdf(_file):  # Streamlit 파일 업로더가 준 파일객체를 받아 처리

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:  # OS가 임시 파일을 생성(나중에 우리가 지울 수 있도록 delete=False)
        tmp_file.write(_file.getvalue())  # 업로드된 파일의 바이트를 임시 파일에 기록
        tmp_file_path = tmp_file.name  # 임시 파일 경로 문자열을 확보
        # PDF 파일 업로드 완료 -> 이제 로딩
        loader = PyPDFLoader(file_path=tmp_file_path)  # 임시 경로를 사용해 로더 생성
        pages = loader.load_and_split()  # 페이지별 문서 리스트로 분해
    return pages  # 페이지 리스트 반환


@st.cache_resource
def create_vector_store(_docs):  # 문서 리스트를 받아 벡터DB(Chroma)를 메모리 상에서 생성
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # 500자 단위로, 50자 겹치게 분할
    split_docs = text_splitter.split_documents(_docs)  # 문서들을 작은 청크들로 나눔
    # persist_directory = "./chroma_pdf_db"  # 디스크에 영구저장하려면 경로를 설정(여기선 메모리 사용이라 주석)
    vectorstore = Chroma.from_documents(  # 분할된 문서를 임베딩하여 Chroma에 저장(메모리)
        split_docs,
        OpenAIEmbeddings(model='text-embedding-3-small'),  # 가벼운 임베딩 모델 선택(비용/속도/정확도 균형)
        # persist_directory=persist_directory  # 디스크에 저장하려면 주석 해제
    )
    return vectorstore  # 검색(retrieval)에 사용할 벡터 스토어 반환


# (옵션) 이미 디스크에 만들어둔 ChromaDB를 재사용하려는 경우 불러오는 함수
@st.cache_resource
def get_vectorstore():  # 영구저장된 Chroma 인스턴스를 경로에서 로드
    persist_directory = "./chroma_db"  # 미리 만들어 둔 DB 폴더 경로(프로젝트 루트 기준)
    print(persist_directory)  # 디버그용 출력(서버 로그에서 확인 가능)
    # if os.path.exists(persist_directory):  # 경로 확인 로직(현재는 단순화를 위해 주석 처리)
    return Chroma(  # 해당 경로의 DB를 로드
        persist_directory=persist_directory,
        embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')  # 동일한 임베딩 설정 필요
    )
    # else:
    #     return 0  # (단순화를 위해 미사용)


def format_docs(docs):  # 검색된 여러 문서를 하나의 큰 문자열로 합치는 간단한 유틸
    return "\n\n".join(doc.page_content for doc in docs)  # 문서들의 본문을 두 줄 공백으로 이어붙임


# PDF 문서 로드-벡터 DB-검색기-대화 이력까지 모두 합친 RAG 체인 초기화 함수(기존 DB 재사용 모드)
@st.cache_resource
def initialize_components(selected_model):  # 선택한 LLM 모델명을 받아 체인을 구성
    # file_path = r"../data/"  # (예시 경로) 직접 파일 로딩이 필요하면 사용
    # file_path = r"C:/Users/Jay/PycharmProjects/test_ai/input3.pdf"  # (예시 경로)
    # pages = load_and_split_pdf(file_path)  # (예시) 파일에서 직접 로드하려면 사용
    vectorstore = get_vectorstore()  # 디스크의 Chroma DB를 로드
    retriever = vectorstore.as_retriever()  # 벡터 DB를 질의용 검색기로 변환

    # 사용자의 최신 질문이 과거 대화 맥락을 가리킬 수 있으므로, 우선 '독립형 질문'으로 재구성하는 시스템 프롬프트
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""  # 영어 유지: 모델 프롬프트 관례

    contextualize_q_prompt = ChatPromptTemplate.from_messages(  # 위 시스템 지시 + 과거이력 + 현재 입력을 하나의 메시지 세트로 묶음
        [
            ("system", contextualize_q_system_prompt),  # 시스템 역할 메시지(규칙/지시)
            MessagesPlaceholder("history"),  # 과거 대화 이력이 여기에 채워짐
            ("human", "{input}"),  # 사용자의 현재 질문(포맷팅으로 주입)
        ]
    )

    # 실제 답변용 시스템 프롬프트(검색된 문맥 기반으로만 답하도록 엄격히 지시)
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    Please say answer only based pieces of retrieved context.\
    If you don't know the answers, Please say you don't know. \
    Keep the answer perfect. please use imogi with the answer.\
    대답은 한국어로 하고, 존댓말을 써줘.\
    {context}"""  # 한국어로 공손하게 답하고, 이모지도 쓰라는 지시 포함

    qa_prompt = ChatPromptTemplate.from_messages(  # 최종 답변 프롬프트 구성(시스템+이력+현재 질문)
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 토큰제한 걱정 메모(실제 제어는 model, chunking, k값 등으로 조절)
    llm = ChatOpenAI(model=selected_model, temperature=0)  # OpenAI 챗 모델을 0온도(창의성 최소)로 설정: 문서 기반 사실위주 답변 유도

    # 대화 이력을 고려해, 먼저 질문을 독립형으로 가다듬고(retriever), 그 질문으로 문서를 검색하는 검색기 구성
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)  # 검색된 문서를 프롬프트에 "붙여넣어" 답변하는 체인
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)  # 검색과 답변 체인을 한데 묶은 최종 체인

    return rag_chain  # UI에서 이 체인을 호출하여 질의응답을 수행


# 아래는 다양한 검색 전략 예시(현재 미사용). 필요 시 주석 해제하여 실험.
# retriever = vectorstore.as_retriever(
#     search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.6})


# ChatGPT 단독 모드(내 자료 검색 없이)로 간단 대화하는 화면 로직
# @st.cache_resource  # (캐시 불필요해서 주석)
def initial_not_select(selected_model):  # 모델명을 받아 ChatGPT 대화만 하는 모드
    if "messages" not in st.session_state:  # 세션에 대화 기록이 없다면
        st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요 챗봇입니다. 무엇을 도와드릴까요?"}]  # 첫 인사 저장
    print(st.session_state)  # 디버그용 출력
    for msg in st.session_state.messages:  # 지금까지의 대화를 화면에 순서대로 출력
        st.chat_message(msg["role"]).write(msg["content"])  # 역할에 따라 말풍선 형태로 표시
    # prompt = ""  # (사용 안 함)
    if prompt := st.chat_input(key=1):  # 하단 입력창에서 사용자가 질문을 입력하면
        client = OpenAI()  # OpenAI 공식 SDK 클라이언트 생성
        st.session_state.messages.append({"role": "user", "content": prompt})  # 대화 기록에 사용자 메시지 추가
        st.chat_message("user").write(prompt)  # 사용자 말풍선 출력
        response = client.chat.completions.create(model=selected_model, messages=st.session_state.messages)  # Chat Completions API 호출
        msg = response.choices[0].message.content  # 모델의 답변 텍스트만 추출
        st.session_state.messages.append({"role": "assistant", "content": msg})  # 기록에 답변 추가
        st.chat_message("assistant").write(msg)  # 화면에 답변 출력


@st.cache_resource
def chaining(_pages, selected_model):  # 업로드한 PDF를 즉석에서 벡터DB로 만들어 사용하는 모드
    vectorstore = create_vector_store(_pages)  # 문서에서 벡터 스토어 생성(메모리)
    retriever = vectorstore.as_retriever()  # 검색기로 변환
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})  # 같은 의미(예시)

    # 최신 질문을 독립형으로 재구성하기 위한 시스템 프롬프트(위와 동일 개념)
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
       which might reference context in the chat history, formulate a standalone question \
       which can be understood without the chat history. Do NOT answer the question, \
       just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(  # 시스템 지시 + 과거이력 + 현재 입력 구성
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 이 부분의 시스템 프롬프트는 팀 취향대로 바꿀 수 있습니다(존댓말/이모지 지시 등 포함).
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    Please say answer only based pieces of retrieved context.\
    If you don't know the answers, Please say you don't know. \
    Keep the answer perfect. please use imogi with the answer.\
    대답은 한국어로 하고, 존댓말을 써줘.\
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(  # 최종 답변 프롬프트(여기선 history 없이 간단 구성)
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=selected_model, temperature=0)  # 사실 위주 응답을 위해 temperature=0
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)  # 대화 이력 인지 검색기
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)  # 문서를 통째로 프롬프트에 결합하는 체인
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)  # 검색+생성 결합
    return rag_chain  # 호출 측에서 invoke로 실행


# 세션 상태에 로그인 플래그가 없으면 기본값 False로 시작
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False  # 아직 로그인 안 됨


# 매우 단순한 로그인 함수(실서비스 용도가 아닌 데모 수준). 입력값이 secrets의 id와 같으면 통과.
def login(id_in):  # 사용자가 입력한 아이디(id_in)를 받아 검사
    if id == id_in:  # 미리 저장해 둔 시크릿과 동일하면
        st.session_state["logged_in"] = True  # 로그인 성공 플래그 설정
        st.success("login succes")  # 성공 메시지(오타는 유지)
        st.rerun()  # 현재 앱을 재실행하여 로그인 이후 화면을 즉시 보여줌
    else:
        st.error(("Wrong ID and Password"))  # 실패 메시지(비밀번호는 실제론 검증하지 않음)


# 로그인 여부에 따라 다른 화면을 렌더링
if not st.session_state["logged_in"]:  # 아직 로그인 전이면
    st.title("Login1")  # 상단 제목 표시
    id_in = st.text_input("ID")  # 아이디 입력창
    pw = st.text_input("password", type="password")  # 비밀번호 입력창(가려서 표시). 실제 검증은 안 함
    if st.button("Login"):  # 로그인 버튼을 누르면
        cache_clear()  # 캐시를 비워 깔끔한 상태로 시작
        print(id_in)  # 디버그용 출력
        print(id)  # 디버그용 출력(운영에선 민감정보 로그출력 금지!)
        login(id_in)  # 로그인 시도
else:

        # Streamlit UI (로그인 이후 보여줄 본 기능 화면)
    st.header("항공통신소 Q&A 챗봇 💬")  # 앱의 메인 헤더
    selection = st.selectbox("ChatGpt,기존 Database(노하우 등), PDF ", ("ChatGpt", "Database", "PDF"))  # 사용할 모드 선택
    option = st.selectbox("Select GPT Model", ("gpt-4.1-mini", "gpt-4.1"))  # 사용할 OpenAI 모델 선택
    # st.slider('몇살인가요?', 0, 130, 25)  # (예시 UI) 현재 미사용
    # halu_t = st.slider("기존 문서로 답변: 0, 창의력 추가 답변: 1", 0.0,1.0,(0.0))  # (예시)
    # halu = st.selectbox("기존 문서로 답변: 0, 창의력 추가 답변: 1",("0","0.5","1"))  # (예시)

    # halu = str(halu_t)  # (예시)

    # print(halu_t)  # (예시)
    if selection == "ChatGpt":  # ChatGPT 단독 모드 선택 시
        cache_clear()  # 이전 리소스 캐시를 지워 모드 전환 시 충돌 방지
        initial_not_select(option)  # 단독 대화 UI 실행

    if selection == "PDF":  # PDF 기반 RAG 모드 선택 시
        # cache_clear()  # (필요에 따라 캐시 초기화)
        uploaded_file = st.file_uploader("PDF 기반 답변", type=["pdf"], accept_multiple_files=True)  # 여러 PDF 업로드
        for file in uploaded_file:  # 업로드된 각 파일에 대해
            pages = load_pdf(file)  # 임시파일로 저장 후 페이지 리스트 로드
            print(pages)  # 디버그용(페이지 객체 목록)
            print(type(pages))  # 디버그용(자료형)
        try:
            rag_chain = chaining(pages, option)  # 방금 로드한 페이지들로 벡터DB를 만들고 RAG 체인 구성
            # print(rag_chain)  # 디버그용
            chat_history = StreamlitChatMessageHistory(key="chat_messages")  # 세션에 대화 이력을 보관/복원하는 객체
            if "messages" not in st.session_state:  # (별도 대화 기록 키를 쓰는 경우 초기화)
                st.session_state["messages"] = [{"role": "assistant",
                                                 "content": "무엇이든 물어!"}]  # 간단한 안내 말풍선

            conversational_rag_chain = RunnableWithMessageHistory(  # 체인에 "대화 이력" 기능을 연결해, 매 턴 히스토리가 전달되게 함
                rag_chain,
                lambda session_id: chat_history,  # 세션ID로 히스토리 객체를 반환하는 콜백(여기선 고정)
                input_messages_key="input",  # 체인에 입력 프롬프트가 들어갈 키 이름
                history_messages_key="history",  # 과거 대화가 주입될 키 이름
                output_messages_key="answer",  # 체인 결과에서 답변이 담기는 키 이름
            )

            for msg in chat_history.messages:  # 지금까지의 히스토리를 UI에 표시
                st.chat_message(msg.type).write(msg.content)

            if prompt_message := st.chat_input("Your question"):  # 사용자 질문 입력
                st.chat_message("human").write(prompt_message)  # 사용자 말풍선 표시
                with st.chat_message("ai"):  # AI 말풍선 컨텍스트 내부에서
                    with st.spinner("Thinking..."):  # 처리 중 스피너 표시
                        config = {"configurable": {"session_id": "any"}}  # 히스토리 식별용 세션ID(간단히 고정 문자열 사용)
                        response = conversational_rag_chain.invoke(  # 체인을 한 번 실행하여 결과 받기
                            {"input": prompt_message},
                            config)

                        answer = response['answer']  # 체인이 반환한 딕셔너리에서 답변 텍스트만 추출
                        st.write(answer)  # 답변 출력
                        # with st.expander("참고 문서 확인"):  # 참고한 문서(근거)도 보여주고 싶다면 주석 해제
                        #     for doc in response['context']:
                        #         st.markdown(doc.metadata['source'], help=doc.page_content)
        except:  # 업로드/파싱/체인 구성 중 오류가 나면(예: 파일 미선택)
            st.header("📚 PDF 업로드 해주세요!")  # 사용자에게 업로드 요청 안내
            # st.header("한번에 여러 파일 업로드")  # (추가 안내 예시)

    elif selection == "Database":  # 기존에 구축해 둔 로컬 Chroma DB로 답변하는 모드
        # cache_clear()  # (필요에 따라 캐시 초기화)
        rag_chain = initialize_components(option)  # DB 로드 + RAG 체인 준비
        chat_history = StreamlitChatMessageHistory(key="chat_messages")  # 대화 이력 관리 객체

        conversational_rag_chain = RunnableWithMessageHistory(  # 히스토리 포함 체인 래퍼
            rag_chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )
        print(st.session_state)  # 디버그용 출력
        if "messages" not in st.session_state:  # 없으면 간단한 안내 메시지로 초기화
            st.session_state["messages"] = [{"role": "assistant",
                                             "content": "무엇이든 물어보세요!"}]

        for msg in chat_history.messages:  # 이전 대화 표시
            st.chat_message(msg.type).write(msg.content)

        if prompt_message := st.chat_input("Your question"):  # 질문 입력
            st.chat_message("human").write(prompt_message)  # 사용자 말풍선
            with st.chat_message("ai"):  # AI 말풍선
                with st.spinner("Thinking..."):  # 처리 중 표시
                    config = {"configurable": {"session_id": "any"}}  # 세션ID 설정(간단 고정)
                    response = conversational_rag_chain.invoke(  # 체인 실행
                        {"input": prompt_message},
                        config)

                    answer = response['answer']  # 답변 텍스트만 추출
                    st.write(answer)  # 화면 출력
                    # with st.expander("참고 문서 확인"):  # 근거 문서 UI (원하면 사용)
                    #     for doc in response['context']:
                    #         st.markdown(doc.metadata['source'], help=doc.page_content)  # 문서 출처/내용 힌트
