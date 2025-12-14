from typing import List
from langchain_core.documents import Document


class EnsembleRetriever:
    """여러 retriever의 결과를 가중치 기반으로 결합하는 앙상블 리트리버"""
    
    def __init__(self, retrievers: List, weights: List[float]):
        self.retrievers = retrievers
        self.weights = weights
    
    def invoke(self, query: str) -> List[Document]:
        """여러 retriever의 결과를 가중치 기반으로 결합"""
        all_docs = []
        doc_scores = {}
        
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            
            # 각 문서에 가중치 적용
            for i, doc in enumerate(docs):
                doc_id = hash(doc.page_content)
                # 순위 기반 스코어 (상위일수록 높은 점수)
                score = weight * (len(docs) - i) / len(docs)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]['score'] += score
                else:
                    doc_scores[doc_id] = {'doc': doc, 'score': score}
        
        # 스코어 기준으로 정렬
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs]

