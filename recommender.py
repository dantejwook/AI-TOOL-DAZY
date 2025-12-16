# modules/recommender.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


def recommend_by_cosine(new_doc_vector, existing_vectors, top_n=3):
    """
    벡터 유사도 기반 추천 문서 리스트 반환.
    """
    similarities = cosine_similarity([new_doc_vector], existing_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return top_indices, similarities[top_indices]


def explain_document_similarity(target_doc: str, related_docs: list[tuple[str, str]]) -> str:
    """
    GPT에게 기준 문서와 관련 문서들 간의 주제 유사성 설명 요청 (JSON 형식).
    related_docs: (doc_id, text)
    """
    system_prompt = """
당신은 문서 간의 공통 주제를 간단히 설명하는 분석가입니다.

규칙:
- 문서 내용을 요약하거나 재작성하지 마세요.
- 공통된 주제 또는 관점만 한 문장으로 설명하세요.
- 각 문서당 한 문장만 작성하세요.
- 출력은 반드시 JSON 형식만 사용하세요.
"""

    related_json = "\n".join([f"- {doc_id}: {text}" for doc_id, text in related_docs])
    user_prompt = f"""
기준 문서:
{target_doc}

연관 문서 목록:
{related_json}

작업 지시:
각 연관 문서가 기준 문서와 왜 함께 읽으면 좋은지 한 문장으로 설명하세요.

출력 형식 (JSON만):

{{
  "recommendations": [
    {{
      "document_id": "",
      "reason": ""
    }}
  ]
}}
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
