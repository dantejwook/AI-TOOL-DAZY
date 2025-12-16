# modules/embedder.py

import openai
from config import OPENAI_API_KEY, EMBED_MODEL
import json
import os

openai.api_key = OPENAI_API_KEY

EMBEDDING_STORE_PATH = "data/embeddings.json"
METADATA_STORE_PATH = "data/metadata.json"


def get_embedding(text: str) -> list[float]:
    """
    OpenAI embedding 모델을 이용하여 임베딩 벡터 생성
    """
    response = openai.Embedding.create(
        model=EMBED_MODEL,
        input=text
    )
    return response['data'][0]['embedding']


def load_store(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_store(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_and_store_embeddings(chunks: list[str], doc_id: str):
    """
    문서 chunk 리스트를 임베딩 처리하고 저장
    저장 위치:
        - data/embeddings.json
        - data/metadata.json
    """
    embedding_store = load_store(EMBEDDING_STORE_PATH)
    metadata_store = load_store(METADATA_STORE_PATH)

    chunk_vectors = [get_embedding(chunk) for chunk in chunks]
    avg_vector = [sum(x) / len(x) for x in zip(*chunk_vectors)]

    embedding_store[doc_id] = avg_vector
    metadata_store[doc_id] = {
        "chunk_count": len(chunks)
    }

    save_store(EMBEDDING_STORE_PATH, embedding_store)
    save_store(METADATA_STORE_PATH, metadata_store)

    return avg_vector
