# modules/clustering.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple


def determine_best_k(vectors: List[List[float]], k_range: Tuple[int, int] = (2, 10)) -> int:
    """
    Elbow 또는 Silhouette 방법으로 최적의 K 값을 자동 결정
    """
    best_k = k_range[0]
    best_score = -1

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(vectors)
        score = silhouette_score(vectors, labels)
        if score > best_score:
            best_k = k
            best_score = score

    return best_k


def cluster_embeddings(
    vectors: List[List[float]],
    doc_ids: List[str],
    auto_k: bool = True,
    fixed_k: int = 5
) -> Dict[str, int]:
    """
    문서 임베딩 벡터를 KMeans로 클러스터링하고,
    각 문서의 클러스터 ID를 반환합니다.

    Parameters:
    - vectors: List of embedding vectors
    - doc_ids: 문서 고유 식별자 (filename 등)
    - auto_k: 최적 K 자동 탐색 여부
    - fixed_k: auto_k=False일 때 사용할 고정 클러스터 수

    Returns:
    - dict {doc_id: cluster_id}
    """

    X = np.array(vectors)

    k = determine_best_k(X) if auto_k else fixed_k

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    clustered = {doc_id: int(cluster_id) for doc_id, cluster_id in zip(doc_ids, labels)}
    return clustered


def merge_small_clusters(
    cluster_map: Dict[str, int],
    min_size: int = 2
) -> Dict[str, int]:
    """
    너무 작은 클러스터를 다른 클러스터에 병합 처리

    Parameters:
    - cluster_map: {doc_id: cluster_id}
    - min_size: 최소 문서 수 기준

    Returns:
    - 병합된 cluster_map
    """
    from collections import defaultdict, Counter

    cluster_counter = Counter(cluster_map.values())
    small_clusters = [cid for cid, count in cluster_counter.items() if count < min_size]

    # 병합: 작은 클러스터 문서를 가장 큰 클러스터에 편입
    largest_cluster = cluster_counter.most_common(1)[0][0]

    new_cluster_map = cluster_map.copy()
    for doc_id, cluster_id in cluster_map.items():
        if cluster_id in small_clusters:
            new_cluster_map[doc_id] = largest_cluster

    return new_cluster_map
