# modules/gpt_analyzer.py

import openai
from config import OPENAI_API_KEY, GPT_ANALYZER_MODEL, GPT_STRUCTURER_MODEL

openai.api_key = OPENAI_API_KEY


def summarize_cluster(document_texts: list[str]) -> dict:
    """
    클러스터 내 문서들을 GPT에게 전달하여,
    주제, 요약, 키워드 추출 (gpt-5-nano).
    """
    joined_text = "\n\n".join(document_texts)

    system_prompt = """
당신은 여러 문서를 분석하여 공통된 의미를 정리하는 정보 분석가입니다.

규칙:
- 사고 과정이나 분석 이유를 절대 설명하지 마세요.
- 개별 문서를 직접 언급하지 마세요.
- 여러 문서에 공통적으로 나타나는 핵심 의미만 추출하세요.
- 간결하고 명확하게 작성하세요.
- 출력은 반드시 JSON 형식만 사용하세요.
"""

    user_prompt = f"""
아래는 동일한 의미적 클러스터에 속한 여러 문서의 내용입니다.

문서 내용:
{joined_text}

작업 지시:
1. 이 문서 묶음을 대표하는 클러스터 주제를 하나 생성하세요. (최대 12단어)
2. 클러스터 전체를 요약하는 문장을 3~5문장으로 작성하세요.
3. 클러스터를 가장 잘 설명하는 핵심 키워드 5~8개를 추출하세요.

출력 형식 (JSON만):

{{
  "cluster_topic": "",
  "cluster_summary": "",
  "keywords": []
}}
"""

    response = openai.ChatCompletion.create(
        model=GPT_ANALYZER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content


def generate_readme(document_texts: list[str]) -> str:
    """
    여러 문서를 GPT-3.5로 구조 요약 및 README 마크다운 생성.
    """
    joined_text = "\n\n".join(document_texts)

    system_prompt = """
당신은 여러 문서를 구조적으로 요약하여 README.md 형태로 만드는 마크다운 정리 도우미입니다.

규칙:
- 목록, 제목, 소제목 형태로 작성하세요.
- 너무 자세한 내용은 생략하고, 전체 구조만 잡아주세요.
- 출력은 반드시 마크다운 형식이어야 합니다.
"""

    user_prompt = f"""
다음 문서들을 읽고 전체를 요약한 README.md 파일을 생성하세요:

{joined_text}
"""

    response = openai.ChatCompletion.create(
        model=GPT_STRUCTURER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content
