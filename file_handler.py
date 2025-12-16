# modules/file_handler.py

import os
import io
import re
from PyPDF2 import PdfReader
import tiktoken
from typing import List


SUPPORTED_EXTENSIONS = [".pdf", ".md", ".txt"]
CHUNK_TOKEN_SIZE = 500
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def load_file(file) -> str:
    """
    업로드된 파일에서 텍스트 추출
    """
    filename = file.name
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".pdf":
        reader = PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text

    elif ext in [".md", ".txt"]:
        content = file.read()
        return content.decode("utf-8")

    else:
        raise ValueError(f"지원하지 않는 확장자: {ext}")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def split_chunks(text: str, max_tokens: int = CHUNK_TOKEN_SIZE) -> List[str]:
    """
    텍스트를 max_tokens 기준으로 chunk 분할
    """
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        tentative = current_chunk + " " + sentence if current_chunk else sentence
        if count_tokens(tentative) <= max_tokens:
            current_chunk = tentative
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
