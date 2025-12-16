# app.py

import streamlit as st
import os
from modules import file_handler, embedder, clustering, gpt_analyzer, recommender
import json

EMBED_PATH = "data/embeddings.json"
META_PATH = "data/metadata.json"

st.set_page_config(page_title="📄 문서 분석 및 추천", layout="wide")
st.title("📄 문서 의미 분석 및 추천 플랫폼")

if "doc_texts" not in st.session_state:
    st.session_state.doc_texts = {}
if "doc_vectors" not in st.session_state:
    st.session_state.doc_vectors = {}

# STEP 0: 파일 업로드
uploaded_files = st.file_uploader("문서를 업로드하세요 (.pdf, .md, .txt)", type=["pdf", "md", "txt"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        doc_id = file.name
        text = file_handler.load_file(file)
        chunks = file_handler.split_chunks(text)
        avg_vector = embedder.process_and_store_embeddings(chunks, doc_id)

        st.session_state.doc_texts[doc_id] = text
        st.session_state.doc_vectors[doc_id] = avg_vector

    st.success(f"{len(uploaded_files)}개 문서 처리 완료 ✅")

# STEP 3~5: 분석 실행 버튼
if st.button("🚀 의미 기반 분석 실행"):
    if not st.session_state.doc_vectors:
        st.warning("먼저 문서를 업로드하세요.")
    else:
        st.subheader("📊 클러스터링 결과")

        # 클러스터링
        doc_ids = list(st.session_state.doc_vectors.keys())
        vectors = list(st.session_state.doc_vectors.values())
        cluster_map = clustering.cluster_embeddings(vectors, doc_ids)
        cluster_map = clustering.merge_small_clusters(cluster_map)

        # 클러스터별 문서 그룹
        from collections import defaultdict
        clusters = defaultdict(list)
        for doc_id, cluster_id in cluster_map.items():
            clusters[cluster_id].append(doc_id)

        for cluster_id, doc_list in clusters.items():
            st.markdown(f"### 📁 클러스터 {cluster_id}")
            texts = [st.session_state.doc_texts[doc_id] for doc_id in doc_list]
            result_json = gpt_analyzer.summarize_cluster(texts)

            try:
                result = json.loads(result_json)
                st.write(f"📌 주제: **{result['cluster_topic']}**")
                st.write(f"📝 요약: {result['cluster_summary']}")
                st.write("🔑 키워드:", ", ".join([f"`{kw}`" for kw in result["keywords"]]))
            except Exception as e:
                st.error("GPT 응답 파싱 오류:", result_json)

            st.write("📄 문서 목록:")
            for doc_id in doc_list:
                st.markdown(f"- {doc_id}")

        # STEP 5: 추천 예시
        st.subheader("📚 유사 문서 추천 예시")

        target_doc = doc_ids[0]
        target_vec = st.session_state.doc_vectors[target_doc]
        other_vectors = [vec for i, vec in enumerate(vectors) if doc_ids[i] != target_doc]
        other_ids = [doc_ids[i] for i in range(len(doc_ids)) if doc_ids[i] != target_doc]

        top_idxs, _ = recommender.recommend_by_cosine(target_vec, other_vectors)
        top_docs = [other_ids[i] for i in top_idxs]
        pairs = [(doc_id, st.session_state.doc_texts[doc_id]) for doc_id in top_docs]
        explanation_json = recommender.explain_document_similarity(st.session_state.doc_texts[target_doc], pairs)

        try:
            explanation = json.loads(explanation_json)
            for rec in explanation["recommendations"]:
                st.markdown(f"🔗 **{rec['document_id']}**: {rec['reason']}")
        except:
            st.error("추천 결과 파싱 오류")
