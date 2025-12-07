
'''
의학 데이터 전처리 & 임베딩 & 벡터스토어 코드
'''


import os, json, glob
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangChain 관련 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma 
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


# 경고 메세지 무시
import warnings
warnings.filterwarnings("ignore")

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

if not os.environ.get('OPENAI_API_KEY'):
    raise ValueError('.env 확인하세요. key가 없습니다')




#1. 의학지식 데이터 전처리
print("\n" + "=" * 30)
print("의학지식 데이터 로드 및 전처리")
print("=" * 30)

paths = [
    r".\3차 프로젝트\data\말뭉치\TS_말뭉치데이터_내과",
    r".\3차 프로젝트\data\말뭉치\TS_말뭉치데이터_안과",
    r".\3차 프로젝트\data\말뭉치\TS_말뭉치데이터_외과",
    r".\3차 프로젝트\data\말뭉치\TS_말뭉치데이터_치과",
    r".\3차 프로젝트\data\말뭉치\TS_말뭉치데이터_피부과"]
     

docs = []

#json -> Document 변환
# 각 경로에 대해 반복
for path in paths:
    print(f"처리 중인 경로: {path}")
    
    for file_path in glob.glob(os.path.join(path, "**", "*.json"), recursive=True):
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        disease = data.get("disease", "") or ""

        # 문서 내용: 질병
        page_content = disease
     
        # 메타데이터 구성
        meta = {
            "title": data.get("title", ""),
            "author": data.get("author", None),
            "publisher": data.get("publisher", None),
            "department": data.get("department", None),

            #기존 메타데이터에 source_type과 source_path 추가
            "source_type": "medical data",
            "source_path": path,  # 어느 경로에서 왔는지 추가
        }

        docs.append(Document(page_content=page_content, metadata=meta))


print(f"총 {len(docs)}개 문서를 로드했습니다.")
print(docs[0].page_content[:300])
print(docs[0].metadata)

# 2.질의응답 데이터 전처리
print("\n" + "=" * 30)
print("질의응답 데이터 로드 및 전처리")
print("=" * 30)

paths_qa = [
    r".\3차 프로젝트\data\qa\TL_질의응답데이터_내과",
    r".\3차 프로젝트\data\qa\TL_질의응답데이터_안과",
    r".\3차 프로젝트\data\qa\TL_질의응답데이터_외과",
    r".\3차 프로젝트\data\qa\TL_질의응답데이터_치과",
    r".\3차 프로젝트\data\qa\TL_질의응답데이터_피부과"]
     
docs_qa = []

# 각 경로에 대해 반복
for path_qa in paths_qa:
    print(f"처리 중인 경로: {path_qa}")
    
    for file_path in glob.glob(os.path.join(path_qa, "**", "*.json"), recursive=True):
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        # meta와 qa 추출
        meta_info = data.get("meta", {})
        qa_info = data.get("qa", {})
        
        # page_content: 질문 + 답변을 하나로 합치기
        question = qa_info.get("input", "")
        answer = qa_info.get("output", "")
        
        # Q&A 형태로 구성 (검색 시 더 효과적)
        page_content = f"Q: {question}\n\nA: {answer}"
        
        # metadata: 메타정보 + QA 관련 정보
        metadata = {
            # 기존 메타 정보
            "lifeCycle": meta_info.get("lifeCycle", ""),
            "department": meta_info.get("department", ""),
            "disease": meta_info.get("disease", ""),
            
            # QA 관련 정보
            "question": question,
            "answer": answer,

            #기존 메타데이터에 source_type과 source_path 추가
            "source_type": "qa_data",
            "source_path": path_qa  # path 대신 path_qa 사용
        }

        docs_qa.append(Document(page_content=page_content, metadata=metadata))

print(f"총 {len(docs_qa)}개 문서를 로드했습니다.")
print(docs_qa[0].page_content[:300])
print(docs_qa[0].metadata)

# docs와 docs_qa 합치기
docs.extend(docs_qa)
print(f"\n최종 문서 개수: {len(docs)}개")

# docs.pkl 저장
# import pickle
# with open("final_docs.pkl", "wb") as f:
#     pickle.dump(docs, f)
# print("final_docs.pkl 저장 완료")


# 3. 청킹
print("\n" + "=" * 30)
print("문서 청킹 처리")
print("=" * 30)

# 데이터 타입별 splitter 정의
splitter_map = {
    # 의학 데이터 (긴 설명문)
    "medical_data": RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=['\n\n', '\n', '.', '!', '?', ',', ' ', '']
    ),
    
    # QA 데이터 (질문-답변 쌍)
    "qa_data": RecursiveCharacterTextSplitter(
        chunk_size=800,  # QA는 더 큰 청크로
        chunk_overlap=50,
        separators=['\n\nA:', 'Q:', '\n\n', '\n', '.', ' ', '']
    )
}

# 기본 splitter (매칭되지 않는 경우)
default_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=['\n\n', '\n', '.', ',', ' ', '']
)

chunked_docs = []

print(f"\n청킹 대상 원본 Document 수: {len(docs)}개")

# 각 문서의 source_type에 따라 다른 splitter 적용
for doc in docs:
    source_type = doc.metadata.get("source_type", "")
    
    # 데이터 타입에 맞는 splitter 선택
    if source_type == "medical data":
        splitter = splitter_map["medical_data"]
        print(f"의학 데이터 청킹")
    elif source_type == "qa_data":
        splitter = splitter_map["qa_data"]  
        print(f"QA 데이터 청킹")
    else:
        splitter = default_splitter
        print(f"기본 청킹")
    
    # 청킹 실행
    chunks = splitter.split_documents([doc])
    
    # 청킹된 문서들에 원본 메타데이터 보존 + 청킹 정보 추가
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_method": source_type
        })
    
    chunked_docs.extend(chunks)


print(f" 최종 청킹 결과: {len(chunked_docs)}개 Document")
# 청킹 파일 저장

import pickle  
with open("chunked_docs.pkl", "wb") as f:
    pickle.dump(chunked_docs, f)  
print("chunked_documents.pkl 저장 완료")