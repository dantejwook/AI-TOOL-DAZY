import streamlit as st
from modules import file_handler, embedder, clustering, gpt_analyzer, recommender

def main():
    st.title("📄 문서 의미 분석 및 추천 플랫폼")
    
    # STEP 0: 파일 업로드
    uploaded_files = st.file_uploader("문서를 업로드하세요", type=["pdf", "txt", "md"], accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            text = file_handler.load_file(file)
            chunks = file_handler.split_chunks(text)
            embedding_info = embedder.process_and_store_embeddings(chunks, file.name)
            # 이후 파이프라인 연결

    # 클러스터링, 요약, 추천은 버튼으로 트리거
    if st.button("🚀 의미 기반 분석 실행"):
        # 1. 클러스터링
        # 2. GPT 해석
        # 3. 시각화 및 마크다운 생성
        # 4. 유사 문서 추천
        
        st.success("분석이 완료되었습니다.")
