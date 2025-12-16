# modules/name_classifier.py

import re
from typing import Dict


def classify_by_filename(filename: str) -> Dict[str, str]:
    """
    파일명을 기반으로 soft grouping을 위한 메타데이터 추출

    예: "회의록_2023_3월.pdf" → {"year": "2023", "month": "3", "category": "회의록"}

    반환값:
    {
        "category": "회의록",
        "year": "2023",
        "month": "3"
    }
    """
    base_name = filename.rsplit(".", 1)[0]
    tokens = re.split(r'[_\- ]+', base_name)

    result = {
        "category": None,
        "year": None,
        "month": None
    }

    # category candidates
    categories = ["회의록", "가이드", "정책", "보고서", "전략", "계획"]

    for token in tokens:
        if token in categories and result["category"] is None:
            result["category"] = token

        if re.fullmatch(r"\d{4}", token) and result["year"] is None:
            result["year"] = token

        if token.endswith("월") or re.fullmatch(r"[1-9]|1[0-2]", token):
            month_clean = re.sub(r"월", "", token)
            result["month"] = month_clean

    return result
