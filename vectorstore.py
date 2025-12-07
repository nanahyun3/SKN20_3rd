import os
import warnings
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError('OPENAI_API_KEY not set')
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import pickle
import time



with open("chunked_docs.pkl", "rb") as f:
    final_docs = pickle.load(f)


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 배치 크기 설정 (토큰 제한 고려)
BATCH_SIZE = 100  # 한 번에 처리할 문서 수


first_batch = final_docs[:BATCH_SIZE]

vectorstore = Chroma.from_documents(
    documents=first_batch,
    embedding=embedding_model,
    collection_name="pet_health_qa_system",
    persist_directory="./ChromaDB",
)

print(f"첫 번째 배치 완료: {len(first_batch)}개 문서")

# 나머지 문서들을 배치로 추가
remaining_docs = final_docs[BATCH_SIZE:]
total_batches = len(remaining_docs) // BATCH_SIZE + (1 if len(remaining_docs) % BATCH_SIZE > 0 else 0)

for i in range(0, len(remaining_docs), BATCH_SIZE):
    batch_num = i // BATCH_SIZE + 2  # 첫 번째 배치 다음부터
    batch = remaining_docs[i:i + BATCH_SIZE]
    
    print(f"배치 {batch_num}/{total_batches + 1} 처리 중... ({len(batch)}개 문서)")
    
    try:
        vectorstore.add_documents(batch)
        print(f"배치 {batch_num} 완료!")
        
        # API 호출 제한 방지를 위한 잠시 대기
        time.sleep(1)
        
    except Exception as e:
        print(f"배치 {batch_num} 에러: {e}")
        # 에러 발생 시 더 작은 배치로 재시도
        smaller_batches = [batch[j:j+20] for j in range(0, len(batch), 20)]
        for small_batch in smaller_batches:
            try:
                vectorstore.add_documents(small_batch)
                time.sleep(0.5)
            except Exception as small_e:
                print(f"소 배치 에러: {small_e}")

print("벡터스토어 생성 완료!")
print(f"저장 경로: ./ChromaDB")
print(f"컬렉션명: pet_health_qa_system")