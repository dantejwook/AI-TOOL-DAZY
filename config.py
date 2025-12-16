# config.py

# OpenAI API
OPENAI_API_KEY = "your-openai-api-key"
EMBED_MODEL = "text-embedding-3-small"
GPT_ANALYZER_MODEL = "gpt-5-nano"
GPT_STRUCTURER_MODEL = "gpt-3.5-turbo"

# 텍스트 분할 기준
CHUNK_TOKEN_SIZE = 500

# 임베딩 및 클러스터링 파일 저장 경로
EMBEDDING_STORE_PATH = "data/embeddings.json"
METADATA_STORE_PATH = "data/metadata.json"
CLUSTERING_STORE_PATH = "data/clustering.json"

# 기타 설정
MIN_CLUSTER_SIZE = 2
RECOMMEND_TOP_N = 3
