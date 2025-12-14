
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")


# LangChain 최신 버전 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma 
import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from ensemble import EnsembleRetriever

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_core.documents import Document

load_dotenv()
if not os.environ.get('OPENAI_API_KEY'):
    raise ValueError('OPENAI_API_KEY 없음. .env 확인하세요')
if not os.environ.get('LANGSMITH_API_KEY'):
    raise ValueError('LANGSMITH_API_KEY 없음. env 확인하세요')


'''
LangSmith 연결(env 셋팅 돼있어야 합니다!)
# pip install -U langchain langsmith (한번만 실행)
'''


os.environ["LANGSMITH_TRACING_V2"] = "true" #기본값 false
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"]="pet_rag" #프로젝트 이름
print("LangSmith 연결 완료")




'''
벡터 DB 불러오기
불러올때 생성시 임베딩 모델/컬렉션 이름과 동일해야 합니다!
'''


# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# bge_m3 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},  # GPU 사용시 'cuda'로 변경
    encode_kwargs={'normalize_embeddings': True}  # bge-m3는 정규화 권장
)
#벡터스토어 로드
vectorstore = Chroma(
# persist_directory=r"..\data\ChromaDB_openai", #DB 저장한 경로
# collection_name="pet_health_qa_system",
persist_directory=r"..\data\ChromaDB_bge_m3", #DB 저장한 경로
collection_name="pet_health_qa_system_bge_m3",
embedding_function=embeddings)
print("벡터스토어가 성공적으로 로드되었습니다!")

#컬렉션 확인
client = chromadb.PersistentClient(path=r"..\data\ChromaDB_bge_m3")
collections = client.list_collections()
print("사용 가능한 컬렉션:", [c.name for c in collections])


#프롬포트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 반려견 질병·증상에 대해 수의학 정보를 제공하는 AI 어시스턴트입니다. 
당신의 답변은 반드시 제공된 문맥(Context)만을 기반으로 해야 합니다.
문맥에 없는 정보는 절대로 추측하거나 생성하지 마세요.

[사용 가능한 정보 유형]
- medical_data: 수의학 서적 또는 논문
- qa_data: 보호자-수의사 상담 기록 (생애주기 / 과 / 질병 태그 포함)

[할루시네이션 방지 규칙]
1. 문맥에 없는 정보는 사용하지 마세요.
2. 관련 정보가 없다면 “해당 질문과 관련된 문서를 찾지 못했습니다. ”라고 답변하세요.
3. 여러 문서 제공시, 실제로 답변에 사용한 문서만 출처 명시하세요.
4. **질문에 합당한 답변만 제공하세요. 거짓 정보나 불필요한 정보는 제외하세요.**

[응답 규칙]
- 사용자 질문을 2~3줄로 요약한다.
- 문맥에서 확인된 가능한 원인을 구체적으로 설명한다. (문맥에 없다면 “문서에 해당 정보가 없습니다”라고 쓴다)
- 집에서 가능한 안전한 관리 방법 2~3개 제안한다. (문맥에 없다면  “문서에 해당 정보가 없습니다”라고 쓴다)
- 어떤 증상이 응급인지 문맥 기반으로 설명한다.
- 병원 방문이 필요한 시기를 구체적으로 안내한다. (문맥에 없다면 “문서에 해당 정보가 없습니다”라고 쓴다)
- 마지막 줄에 반드시 대답 생성에 사용한 모든 문서의 출처를 명시한다
  • 서적 출처: 책 제목 / 저자 / 출판사
  • QA 출처: 생애주기 / 과 / 질병

[전체 톤]
- 공손한 존댓말
- 보호자를 안심시키되, 필요한 부분은 명확하게 안내하는 수의사 상담 톤
         

[출력 형식]
-질문 요약:
-가능한 원인:
-집에서 관리 방법:
-병원 방문 시기:
-출처(참고한 모든 문서)

"""),
("human",
"""
문맥: {context}

사용자 질문: {question}
""")
    ])


# rewrite 프롬프트 
# 사용자의 질문을 키워드 중심으로 정리해 llm 전달 (검색 최적화된 형태로 질문 바꿔줌) 
rewrite_prompt= PromptTemplate.from_template(
    '''
    다음 질문을 검색에 더 적합한 형태로 변환해 주세요.
    키워드 중심으로 명확하게 바꿔주세요
    변환된 검색어만 출력하세요

    원본 질문: {question}
    변환된 검색어:
    ''')



#문서 포맷팅 함수
def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        metadata = doc.metadata
        
        # 데이터 유형에 따라 출처 정보 구성
        if metadata.get("source_type") == "qa_data":
            source_info = f"상담기록 - {metadata.get('lifeCycle', '')}/{metadata.get('department', '')}/{metadata.get('disease', '')}"
        else:
            # 수의학 서적의 경우
            source_info = f"서적 - {metadata.get('title', '')}"
            if metadata.get('author'):
                source_info += f" (저자: {metadata['author']})"
            if metadata.get('page'):
                source_info += f" p.{metadata['page']+1}"
        
        formatted_doc = f"""<document>
                            <content>{doc.page_content}</content>
                            <source_info>{source_info}</source_info>
                            <data_type>{metadata.get('source_type', 'unknown')}</data_type>
                            </document>"""
        
        formatted_docs.append(formatted_doc)
    
    return "\n\n".join(formatted_docs)


collection = vectorstore._collection
doc_count = collection.count()
print(f"벡터스토어 총 문서 수: {doc_count}개")

if doc_count == 0:
    raise ValueError("벡터스토어가 비어있습니다. 먼저 문서를 추가해주세요.")

# ChromaDB의 get() 메서드를 사용하여 모든 문서 가져오기
all_data = collection.get(limit=doc_count)

# Document 객체로 변환
bm25_docs = []
if all_data and 'ids' in all_data and len(all_data['ids']) > 0:
    documents = all_data.get('documents', [])
    metadatas = all_data.get('metadatas', [])
    
    for i, doc_id in enumerate(all_data['ids']):
        page_content = documents[i] if i < len(documents) else ""
        metadata = metadatas[i] if i < len(metadatas) else {}
        bm25_docs.append(Document(page_content=page_content, metadata=metadata))

if len(bm25_docs) == 0:
    raise ValueError("벡터스토어에서 문서를 가져올 수 없습니다.")

print(f"BM25 리트리버용 문서 {len(bm25_docs)}개 로드 완료")




# 예시 질문으로 성능 테스트
# 무조건 대답해야만 하는거 , 애매한거, 대답 절대 못해야되는거
query = [
    "강아지 파보바이러스 증상은 무엇인가요?",
    "자견 시기 예방접종 스케줄을 알려주세요",
    # "강아지 슬개골 탈구 치료 방법은 무엇인가요?",
    "노령견이 신부전 진단을 받았는데, 식이관리와 약물치료를 병행해야 하나요?",
    "성견의 피부 알레르기와 외이염이 동시에 있을 때 치료 순서는 어떻게 되나요?",
    # "자견이 설사와 구토를 동시에 하는데 응급상황인지 알려주세요",
    # "10살 된 노령견이 갑자기 밥을 안 먹고 기력이 없는데, 어떤 질환을 의심해야 하나요?",
    "중성화 수술 후 체중이 늘어난 성견의 적절한 운동량과 식이량은 어떻게 조절해야 하나요?",
    "강아지 암 예방을 위한 백신이 있나요?",
    "강아지가 초콜릿을 먹었을 때 어떤 약을 먹이면 되나요?"
]

#유사도 검색 리트리버 
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}, search_type="similarity")

# BM25 리트리버
retriever_bm25 = BM25Retriever.from_documents(
    bm25_docs,
    k1=1.2,        # 용어 빈도 포화 매개변수 (기본값: 1.2)
    b=0.75,        # 문서 길이 정규화 매개변수 (기본값: 0.75) 
    k=5,           # 반환할 문서 수 
    preprocess_func=None  # 전처리 함수 (기본값: None)
)
# 앙상블 리트리버
retriever_ensemble = EnsembleRetriever(
    retrievers=[retriever, retriever_bm25],
    weights=[0.5, 0.5]  # 가중치 합 1
) 

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rewrite_chain =  rewrite_prompt | llm | StrOutputParser()
rag_chain = prompt | llm | StrOutputParser()


print(f"=== 앙상블 검색 결과 ===")

for q in query:
    docs = retriever.invoke(q)
    context = format_docs(docs)
    transformed = rewrite_chain.invoke({'question' : q}) #rewrite_chain의 출력(question 키워드)을 transformed에 저장
    generation = rag_chain.invoke({"context": context, "question": transformed})
    print("-"*30)
    print(f'원본 query : {q}\n')
    print(f'transformed query (핵심 키워드 추출) : {transformed}\n')
    print(f"답변: {generation}\n")

