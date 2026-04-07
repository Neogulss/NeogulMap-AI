import os
import re
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
from collections import OrderedDict

import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import faiss
import numpy as np
import pymysql
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import re
from pathlib import Path

# =========================
# 질문 라우팅 함수
# =========================
SERVICE_INTRO_KEYWORDS = [
    "서비스 소개", "서비스 설명", "서비스설명", "입지너구리 소개", "입지너구리 설명",
    "입지너구리가 뭐야", "입지너구리 뭐야", "입지너구리란", "이 서비스가 뭐야",
    "무슨 서비스", "어떤 서비스", "무슨 기능", "주요 기능", "기능 소개",
    "누구를 위한 서비스", "타겟 사용자", "사용 방법", "어떻게 사용", "플랫폼 소개"
]

POLICY_LOAN_KEYWORDS = [
    "정부지원", "지원정책", "정부지원정책", "보조금", "지원금", "창업지원",
    "소상공인 지원", "대출", "창업대출", "소상공인 대출", "정책자금",
    "신용보증", "보증", "사업자대출", "융자", "이차보전", "상환", "금리", "지원"
]

def classify_query(query: str) -> str:
    q = query.strip().lower()

    # 서비스 소개 우선
    for keyword in SERVICE_INTRO_KEYWORDS:
        if keyword.lower() in q:
            return "service_intro"

    # 정책/대출 관련
    for keyword in POLICY_LOAN_KEYWORDS:
        if keyword.lower() in q:
            return "policy_or_loan"

    return "unsupported"


# =========================
# 서비스 소개 문서를 직접 찾는 함수
# =========================
def find_service_intro_file(data_dir: str) -> Path | None:
    data_path = Path(data_dir)
    for file_path in data_path.rglob("입지너구리_서비스소개.md"):
        return file_path
    return None


def load_service_intro_document(data_dir: str) -> dict | None:
    file_path = find_service_intro_file(data_dir)
    if not file_path:
        return None

    title, chunks = build_chunks_from_markdown(file_path)

    if not chunks:
        return None

    merged_content = "\n\n".join(chunk["content"] for chunk in chunks)

    return {
        "title": title,
        "file_name": file_path.name,
        "section": "# 서비스소개",
        "parent_h1": f"# {title}",
        "parent_h2": None,
        "content": merged_content,
        "expanded_content": merged_content,
        "document_type": "service_intro"
    }

# =========================
# 환경 변수 로드
# =========================
load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-mini")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "text-embedding-3-small"
)
DATA_DIR = os.getenv("DATA_DIR", "./original_data")
INDEX_DIR = os.getenv("INDEX_DIR", "./vector_data")

os.makedirs(INDEX_DIR, exist_ok=True)


# =========================
# DB 연결
# =========================
def get_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor
    )


# =========================
# 텍스트 전처리
# =========================
def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def convert_markdown_links(text: str) -> str:
    """
    [텍스트](URL) -> 텍스트: URL
    """
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1: \2", text)


def strip_code_fences(text: str) -> str:
    """
    ``` ... ``` 는 너무 길면 검색 품질을 해칠 수 있어 간단히 정리
    """
    return re.sub(r"```.*?```", "[CODE_BLOCK]", text, flags=re.DOTALL)


def preprocess_markdown(text: str) -> str:
    text = convert_markdown_links(text)
    text = strip_code_fences(text)
    text = normalize_whitespace(text)
    return text


# =========================
# Chunking
# =========================
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def extract_title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        m = re.match(r"^#\s+(.*)$", line.strip())
        if m:
            return m.group(1).strip()
    return fallback


def split_by_markdown_headers(text: str) -> List[Dict[str, str]]:
    """
    마크다운 헤더 계층 유지
    반환 예:
    [
        {
            "h1": "# 정책명",
            "h2": "## 지원대상",
            "header": "## 지원대상",
            "content": "..."
        },
        ...
    ]
    """
    lines = text.splitlines()

    sections = []
    current_h1 = None
    current_h2 = None
    current_header = "문서 시작"
    current_content = []

    for line in lines:
        line = line.rstrip()

        if re.match(r"^#\s+", line):
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    sections.append({
                        "h1": current_h1,
                        "h2": current_h2,
                        "header": current_header,
                        "content": content
                    })
            current_h1 = line.strip()
            current_h2 = None
            current_header = line.strip()
            current_content = []

        elif re.match(r"^##\s+", line):
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    sections.append({
                        "h1": current_h1,
                        "h2": current_h2,
                        "header": current_header,
                        "content": content
                    })
            current_h2 = line.strip()
            current_header = line.strip()
            current_content = []

        else:
            current_content.append(line)

    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            sections.append({
                "h1": current_h1,
                "h2": current_h2,
                "header": current_header,
                "content": content
            })

    return sections


def split_long_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    너무 긴 텍스트를 문자 기준으로 2차 분할
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = max(0, end - overlap)

    return chunks


def approximate_token_count(text: str) -> int:
    """
    대략적인 토큰 수 추정
    정확한 tokenizer 대신 간단 추정값 사용
    """
    return max(1, math.ceil(len(text) / 3))


def build_chunks_from_markdown(file_path: Path,
                               max_chars: int = 1200,
                               overlap: int = 150) -> Tuple[str, List[Dict]]:
    raw_text = file_path.read_text(encoding="utf-8")
    raw_text = preprocess_markdown(raw_text)

    file_name = file_path.name
    title = extract_title_from_markdown(raw_text, fallback=file_name)

    sections = split_by_markdown_headers(raw_text)

    chunks = []
    chunk_index = 0

    for section in sections:
        h1 = section["h1"] or f"# {title}"
        h2 = section["h2"]
        header = section["header"]
        content = section["content"]

        combined = (
            f"문서명: {title}\n"
            f"파일명: {file_name}\n"
            f"상위섹션: {h1}\n"
            f"하위섹션: {h2 or '없음'}\n"
            f"현재섹션: {header}\n"
            f"내용:\n{content}"
        )

        sub_chunks = split_long_text(
            combined,
            max_chars=max_chars,
            overlap=overlap
        )

        for chunk_text in sub_chunks:
            chunks.append({
                "chunk_index": chunk_index,
                "file_name": file_name,
                "title": title,
                "section": header,
                "parent_h1": h1,
                "parent_h2": h2,
                "content": chunk_text,
                "token_count": approximate_token_count(chunk_text),
                "source_path": str(file_path)
            })
            chunk_index += 1

    return title, chunks


# =========================
# DB 저장
# =========================
def upsert_document_and_chunks(conn, title: str, file_path: Path, chunks: List[Dict]) -> tuple[int, List[int]]:
    """
    - 같은 source_path 문서는 기존 데이터 삭제 후 재삽입
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT DOCUMENT_ID FROM CHATBOT_DOCUMENTS WHERE SOURCE_PATH = %s",
            (str(file_path),)
        )
        row = cursor.fetchone()

        if row:
            document_id = row["DOCUMENT_ID"]
            cursor.execute("DELETE FROM CHATBOT_DATA_CHUNKS WHERE DOCUMENT_ID = %s", (document_id,))
            cursor.execute(
                "UPDATE CHATBOT_DOCUMENTS SET FILE_NAME=%s, TITLE=%s WHERE DOCUMENT_ID=%s",
                (file_path.name, title, document_id)
            )
        else:
            cursor.execute(
                """
                INSERT INTO CHATBOT_DOCUMENTS (FILE_NAME, TITLE, SOURCE_PATH)
                VALUES (%s, %s, %s)
                """,
                (file_path.name, title, str(file_path))
            )
            document_id = cursor.lastrowid

        chunk_ids = []
        for chunk in chunks:
            cursor.execute(
                """
                INSERT INTO CHATBOT_DATA_CHUNKS (
                    DOCUMENT_ID, CHUNK_INDEX, SECTION, PARENT_H1, PARENT_H2, CONTENT, TOKEN_COUNT
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    chunk["chunk_index"],
                    chunk["section"],
                    chunk["parent_h1"],
                    chunk["parent_h2"],
                    chunk["content"],
                    chunk["token_count"]
                )
            )
            chunk_ids.append(cursor.lastrowid)

    conn.commit()
    return document_id, chunk_ids


# =========================
# 임베딩
# =========================
class Embedder:
    def __init__(self, model_name: str):
        self.client = OpenAI()
        self.model_name = model_name

    def embed_texts(self, texts, batch_size=100):
        vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )

            for item in response.data:
                vectors.append(item.embedding)

        return np.array(vectors, dtype="float32")

    def embed_query(self, text: str):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text]
        )

        return np.array([response.data[0].embedding], dtype="float32")


def bm25_tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text)
    return tokens


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


# =========================
# FAISS 인덱스
# =========================
def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_faiss_index(index: faiss.Index, id_map: List[int], index_dir: str):
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)


def load_faiss_index(index_dir: str) -> Tuple[faiss.Index, List[int]]:
    index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "id_map.json"), "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return index, id_map


# =========================
# BM25 저장/로드
# =========================
def build_bm25_corpus(chunks: list[dict]) -> tuple[list[list[str]], list[dict]]:
    corpus_tokens = []
    metadata = []

    for chunk in chunks:
        searchable_text = (
            f"{chunk['title']} "
            f"{chunk['section']} "
            f"{chunk['content']}"
        )
        corpus_tokens.append(bm25_tokenize(searchable_text))
        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "title": chunk["title"],
            "file_name": chunk["file_name"],
            "section": chunk["section"],
            "parent_h1": chunk.get("parent_h1"),
            "parent_h2": chunk.get("parent_h2"),
            "content": chunk["content"],
        })

    return corpus_tokens, metadata


def save_bm25_index(bm25, bm25_docs, index_dir: str):
    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "docs": bm25_docs
        }, f)


def load_bm25_index(index_dir: str):
    with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["docs"]


# =========================
# 전체 인덱싱 파이프라인
# =========================
def index_markdown_files(data_dir: str):
    data_path = Path(data_dir)
    md_files = list(data_path.rglob("*.md"))

    if not md_files:
        print(f"[INFO] markdown 파일이 없습니다: {data_dir}")
        return

    embedder = Embedder(EMBED_MODEL)

    all_vectors = []
    all_chunk_ids = []
    all_bm25_docs = []

    conn = get_connection()
    try:
        for file_path in tqdm(md_files, desc="Indexing markdown files"):
            title, chunks = build_chunks_from_markdown(file_path)

            if not chunks:
                print(f"[WARN] chunk가 생성되지 않았습니다: {file_path}")
                continue

            document_id, chunk_ids = upsert_document_and_chunks(conn, title, file_path, chunks)

            texts = [chunk["content"] for chunk in chunks]
            vectors = embedder.embed_texts(texts)
            vectors = normalize(vectors)

            all_vectors.append(vectors)
            all_chunk_ids.extend(chunk_ids)

            for chunk_id, chunk in zip(chunk_ids, chunks):
                all_bm25_docs.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "title": chunk["title"],
                    "file_name": chunk["file_name"],
                    "section": chunk["section"],
                    "parent_h1": chunk.get("parent_h1"),
                    "parent_h2": chunk.get("parent_h2"),
                    "content": chunk["content"],
                })

        if not all_vectors:
            print("[INFO] 생성된 벡터가 없습니다.")
            return

        stacked_vectors = np.vstack(all_vectors).astype("float32")
        index = build_faiss_index(stacked_vectors)
        save_faiss_index(index, all_chunk_ids, INDEX_DIR)

        corpus_tokens, bm25_docs = build_bm25_corpus(all_bm25_docs)
        bm25 = BM25Okapi(corpus_tokens)
        save_bm25_index(bm25, bm25_docs, INDEX_DIR)

        print(f"[DONE] 총 {len(all_chunk_ids)}개 chunk 인덱싱 완료")
        print(f"[DONE] FAISS 저장 위치: {INDEX_DIR}")

    finally:
        conn.close()


# =========================
# BM25 검색
# =========================
def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    bm25, docs = load_bm25_index(INDEX_DIR)

    query_tokens = bm25_tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = []
    for doc, score in ranked:
        results.append({
            "chunk_id": doc["chunk_id"],
            "document_id": doc["document_id"],
            "title": doc["title"],
            "file_name": doc["file_name"],
            "section": doc["section"],
            "parent_h1": doc["parent_h1"],
            "parent_h2": doc["parent_h2"],
            "content": doc["content"],
            "bm25_score": float(score),
        })

    return results


# =========================
# Reranker
# =========================
class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(
            model_name,
            max_length=512,
            device="cpu"
        )

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10):
        if not candidates:
            return []

        pairs = []
        for item in candidates:
            doc_text = (
                f"문서명: {item['title']}\n"
                f"섹션: {item['section']}\n"
                f"내용:\n{item['content']}"
            )
            pairs.append([query, doc_text])

        scores = self.model.predict(pairs, batch_size=16)

        reranked = []
        for item, score in zip(candidates, scores):
            new_item = dict(item)
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]


# =========================
# RRF
# =========================
def rrf_fusion(semantic_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    merged = {}

    for rank, item in enumerate(semantic_results, start=1):
        chunk_id = item["chunk_id"]
        merged.setdefault(chunk_id, {
            "chunk_id": chunk_id,
            "document_id": item.get("document_id"),
            "title": item.get("title"),
            "file_name": item.get("file_name"),
            "section": item.get("section"),
            "parent_h1": item.get("parent_h1"),
            "parent_h2": item.get("parent_h2"),
            "content": item.get("content"),
            "semantic_rank": None,
            "bm25_rank": None,
            "rrf_score": 0.0
        })
        merged[chunk_id]["semantic_rank"] = rank
        merged[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    for rank, item in enumerate(bm25_results, start=1):
        chunk_id = item["chunk_id"]
        merged.setdefault(chunk_id, {
            "chunk_id": chunk_id,
            "document_id": item.get("document_id"),
            "title": item.get("title"),
            "file_name": item.get("file_name"),
            "section": item.get("section"),
            "parent_h1": item.get("parent_h1"),
            "parent_h2": item.get("parent_h2"),
            "content": item.get("content"),
            "semantic_rank": None,
            "bm25_rank": None,
            "rrf_score": 0.0
        })
        merged[chunk_id]["bm25_rank"] = rank
        merged[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    fused = list(merged.values())
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


# =========================
# 검색용 DB 조회
# =========================
def fetch_chunks_by_ids(chunk_ids: List[int]) -> Dict[int, Dict]:
    if not chunk_ids:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            placeholders = ", ".join(["%s"] * len(chunk_ids))
            sql = f"""
                SELECT
                    c.CHUNKS_ID AS id,
                    c.DOCUMENT_ID AS document_id,
                    c.CHUNK_INDEX AS chunk_index,
                    c.SECTION AS section,
                    c.PARENT_H1 AS parent_h1,
                    c.PARENT_H2 AS parent_h2,
                    c.CONTENT AS content,
                    d.TITLE AS title,
                    d.FILE_NAME AS file_name,
                    d.SOURCE_PATH AS source_path
                FROM CHATBOT_DATA_CHUNKS c
                JOIN CHATBOT_DOCUMENTS d
                  ON c.DOCUMENT_ID = d.DOCUMENT_ID
                WHERE c.CHUNKS_ID IN ({placeholders})
            """
            cursor.execute(sql, chunk_ids)
            rows = cursor.fetchall()
            return {row["id"]: row for row in rows}
    finally:
        conn.close()


def fetch_chunks_by_parent_h1(document_id: int, parent_h1: str) -> List[Dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT
                    c.CHUNKS_ID AS id,
                    c.DOCUMENT_ID AS document_id,
                    c.CHUNK_INDEX AS chunk_index,
                    c.SECTION AS section,
                    c.PARENT_H1 AS parent_h1,
                    c.PARENT_H2 AS parent_h2,
                    c.CONTENT AS content,
                    d.TITLE AS title,
                    d.FILE_NAME AS file_name
                FROM CHATBOT_DATA_CHUNKS c
                JOIN CHATBOT_DOCUMENTS d
                  ON c.DOCUMENT_ID = d.DOCUMENT_ID
                WHERE c.DOCUMENT_ID = %s
                  AND c.PARENT_H1 = %s
                ORDER BY c.CHUNK_INDEX ASC
            """
            cursor.execute(sql, (document_id, parent_h1))
            return cursor.fetchall()
    finally:
        conn.close()


def expand_result_to_parent_context(item: Dict) -> Dict:
    """
    검색 결과 1개를 같은 parent_h1(# 제목) 아래의 모든 하위 섹션으로 확장
    """
    parent_chunks = fetch_chunks_by_parent_h1(
        document_id=item["document_id"],
        parent_h1=item["parent_h1"]
    )

    grouped_sections = OrderedDict()
    for chunk in parent_chunks:
        sec = chunk["section"]
        grouped_sections.setdefault(sec, [])
        grouped_sections[sec].append(chunk["content"])

    merged_sections = []
    for sec, contents in grouped_sections.items():
        merged_sections.append({
            "section": sec,
            "content": "\n\n".join(contents)
        })

    merged_content = "\n\n".join(
        f"[{section['section']}]\n{section['content']}"
        for section in merged_sections
    )

    expanded = dict(item)
    expanded["expanded_content"] = merged_content
    expanded["child_chunks"] = parent_chunks
    expanded["grouped_sections"] = merged_sections
    return expanded


def min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0 for _ in scores]

    return [(s - min_score) / (max_score - min_score) for s in scores]


# =========================
# Hybrid Search
# =========================
def semantic_search(query: str, top_k: int = 10) -> List[Dict]:
    embedder = Embedder(EMBED_MODEL)
    index, id_map = load_faiss_index(INDEX_DIR)

    qvec = embedder.embed_query(query)
    qvec = normalize(qvec)

    scores, indices = index.search(qvec, top_k)

    candidate_ids = []
    score_map = {}

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk_id = id_map[idx]
        candidate_ids.append(chunk_id)
        score_map[chunk_id] = float(score)

    chunk_map = fetch_chunks_by_ids(candidate_ids)

    results = []
    for chunk_id in candidate_ids:
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue
        results.append({
            "chunk_id": chunk_id,
            "document_id": chunk["document_id"],
            "title": chunk["title"],
            "file_name": chunk["file_name"],
            "section": chunk["section"],
            "parent_h1": chunk["parent_h1"],
            "parent_h2": chunk["parent_h2"],
            "content": chunk["content"],
            "semantic_score": score_map[chunk_id],
        })

    return results


def keyword_search_mysql(query: str, limit: int = 10) -> List[Dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT
                    c.CHUNKS_ID AS id,
                    c.DOCUMENT_ID AS document_id,
                    c.CHUNK_INDEX AS chunk_index,
                    c.SECTION AS section,
                    c.CONTENT AS content,
                    MATCH(c.CONTENT) AGAINST(%s IN NATURAL LANGUAGE MODE) AS keyword_score
                FROM CHATBOT_DATA_CHUNKS c
                WHERE MATCH(c.CONTENT) AGAINST(%s IN NATURAL LANGUAGE MODE)
                ORDER BY keyword_score DESC
                LIMIT %s
            """
            cursor.execute(sql, (query, query, limit))
            return cursor.fetchall()
    finally:
        conn.close()


def hybrid_search(query: str,
                  semantic_top_k: int = 20,
                  bm25_top_k: int = 20,
                  fused_top_k: int = 20,
                  rerank_top_k: int = 3) -> list[dict]:
    semantic_results = semantic_search(query, top_k=semantic_top_k)
    bm25_results = bm25_search(query, top_k=bm25_top_k)

    fused = rrf_fusion(semantic_results, bm25_results)
    fused = fused[:fused_top_k]

    reranker = Reranker(
        os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
    )
    reranked = reranker.rerank(query, fused, top_k=rerank_top_k)

    expanded_results = [expand_result_to_parent_context(item) for item in reranked]
    return expanded_results


# =========================
# Answer Generation
# =========================
def build_context_from_results(results: list[dict], max_items: int = 5) -> str:
    """
    검색 결과를 LLM에 넣기 좋은 형태로 정리
    """
    selected = results[:max_items]
    parts = []

    for i, item in enumerate(selected, start=1):
        content_for_answer = item.get("expanded_content", item["content"])

        part = (
            f"[문서 {i}]\n"
            f"문서명: {item['title']}\n"
            f"파일명: {item['file_name']}\n"
            f"섹션: {item['section']}\n"
            f"상위섹션: {item.get('parent_h1', '')}\n"
            f"rerank_score: {item.get('rerank_score', 0.0):.4f}\n"
            f"내용:\n{content_for_answer}\n"
        )
        parts.append(part)

    return "\n\n".join(parts)


def build_single_result_prompt(user_query: str, item: dict) -> str:
    content_for_answer = item.get("expanded_content", item["content"])

    return f"""
너는 소상공인을 위한 한국의 정부지원정책 및 대출 정보를 안내하는 친절한 한국어 챗봇이다.

반드시 아래에 제공된 내용만 바탕으로 답변하라.
없는 내용은 추측하지 말고, 확인이 어려운 부분은 자연스럽게 부족한 점을 설명하라.
숫자, 금액, 날짜, 연령, 비율, 조건은 원래 의미를 바꾸지 말고 유지하라.

답변 원칙:
- 말투는 공고문 요약처럼 딱딱하지 않게, 실제 상담 챗봇처럼 부드럽고 친절하게 작성한다.
- "문서 기준", "문서상", "공고상" 같은 표현은 쓰지 않는다.
- 대신 "안내된 내용을 보면", "지금 보이는 내용으로는", "적혀 있는 내용을 바탕으로 보면" 같은 자연스러운 표현을 사용한다.
- 없는 정보는 "지금 보이는 내용에는 따로 안내되지 않았어요"처럼 부드럽게 말한다.
- 질문과 직접 관련 있는 내용부터 우선해서 설명한다.
- 사용자가 이해하기 쉽게 풀어서 설명하되, 없는 내용을 지어내지는 않는다.
- 답변은 "안내드릴게요", "보여요", "같아요", "확인돼요", "살펴보면" 같은 자연스러운 표현을 적절히 활용하라.

먼저 해야 할 일:
1. 아래 내용이 "지원정책"에 관한 문서인지, "대출상품/대출정보"에 관한 문서인지 먼저 파악하라.
2. 그다음 사용자의 질문 의도가 무엇인지 파악하라.
   - 받을 수 있는지 / 해당되는지
   - 어떤 지원인지 / 어떤 대출인지
   - 지원금액 / 대출한도
   - 신청조건 / 가입대상 세부요건
   - 신청방법 / 이용방법
   - 신청기간
   - 금리 / 상환방식 / 대출종류
   - 유의사항 / 제외대상
   - 전반적인 요약
3. 질문과 직접 관련 있는 항목만 우선해서 설명하라.

문서 유형별 답변 방법:

[지원정책 문서라면]
- 아래와 같은 항목이 보일 수 있다:
  지원대상, 지원내용, 지원금액, 지원조건, 신청방법, 신청기간, 유의사항, 제외대상, 링크 URL
- 하지만 항목명은 문서마다 조금씩 다를 수 있으니, 실제로 보이는 항목을 우선 활용하라.
- 사용자가 받을 수 있는지 물으면, 확인되는 조건만 가지고 자연스럽게 설명하라.
- 가능성 설명은 아래 셋 중 하나의 방향으로 하라.
  - 해당될 가능성이 있어 보여요
  - 어려워 보여요
  - 판단하기는 어려워요

[대출 문서라면]
- 아래와 같은 항목이 보일 수 있다:
  금융회사명, 가입대상 세부요건, 대출한도, 대출종류, 금리방식, 상환방식, 대출금리(평균), 기준금리(평균), 가산금리(평균)
- 하지만 항목명은 문서마다 조금씩 다를 수 있으니, 실제로 보이는 항목을 우선 활용하라.
- 사용자가 받을 수 있는지, 이용할 수 있는지 물으면 가입대상 세부요건, 업종, 신용등급, 지역, 사업자 상태 같은 조건을 중심으로 설명하라.
- 금리를 물으면 대출금리(평균), 기준금리(평균), 가산금리(평균), 금리방식을 우선 설명하라.
- 한도를 물으면 대출한도를 우선 설명하라.
- 상환 관련 질문이면 상환방식과 만기/분할 여부를 우선 설명하라.
- 조건이 일부만 보이면 단정하지 말고, 확인되는 범위까지만 설명하라.

답변 작성 방법:
- 첫 줄: 질문에 대한 짧고 자연스러운 한 줄 답변
- 본문: 질문과 관련된 내용을 중심으로 자연스럽게 설명
- 마지막:
  - 확인해볼 포인트: 1~3개
  - 출처: 문서명, 대표 섹션명

답변에 포함할 내용은 문서 유형에 따라 유연하게 고르라.
예를 들면:
- 지원정책이면 대상, 내용, 금액, 신청조건, 신청방법, 신청기간, 유의사항 중심
- 대출이면 금융회사, 가입대상, 한도, 금리, 상환방식, 대출종류 중심
- 사용자가 묻지 않은 항목은 짧게만 언급하거나 생략할 수 있다.

추가 규칙:
- 링크 URL이 실제로 보이지 않으면 만들어내지 않는다.
- 표처럼 보이는 데이터를 그대로 나열하기보다, 사용자가 이해하기 쉽게 설명형 문장으로 바꿔서 답하라.
- 다만 숫자, 퍼센트, 한도, 기간, 신용등급, 지역 조건은 바꾸지 않는다.
- 여러 항목이 반복되더라도 질문과 가장 관련 있는 정보부터 정리하라.

[사용자 질문]
{user_query}

[문서 메타데이터]
문서명: {item['title']}
파일명: {item['file_name']}
상위섹션: {item.get('parent_h1', '')}
대표 섹션: {item['section']}

[문서 전체 내용]
{content_for_answer}
""".strip()


# 서비스 소개 전용 프롬프트
def build_service_intro_prompt(user_query: str, item: dict) -> str:
    content_for_answer = item.get("expanded_content", item["content"])

    return f"""
너는 입지너구리 서비스 안내 챗봇이다.
반드시 아래 서비스 소개 문서만 근거로 답변하라.

핵심 원칙:
- 문서에 없는 내용은 절대 추측하지 말 것
- 서비스의 목적, 대상 사용자, 주요 기능, 사용 방식, 기대 효과를 중심으로 설명할 것
- 사용자가 묻는 내용이 문서에 없으면 "문서상 확인되지 않음"이라고 답할 것
- 답변은 친절하고 자연스럽게 작성할 것

답변 형식:
- 첫 줄: 사용자의 질문에 대한 한 줄 요약 답변
- 본문: 서비스 소개 문서를 바탕으로 관련 내용을 설명

[사용자 질문]
{user_query}

[서비스 소개 문서]
문서명: {item['title']}
파일명: {item['file_name']}

[문서 내용]
{content_for_answer}
""".strip()


def generate_single_answer(query: str, item: dict, model: str) -> str:
    client = OpenAI()

    if item.get("document_type") == "service_intro":
        prompt = build_service_intro_prompt(query, item)
    else:
        prompt = build_single_result_prompt(query, item)

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text.strip()

# 관련 없는 질문일 때 바로 거절 문구 반환
def build_unsupported_response() -> str:
    return (
        "저희 챗봇 서비스에서는 현재 정부지원정책, 대출 정보, "
        "그리고 서비스 소개 관련 내용만 지원하고 있습니다."
    )

def generate_answers_per_result(query: str, search_results: list[dict], model: str) -> list[dict]:
    answers = []

    for item in search_results:
        answer_text = generate_single_answer(query, item, model)

        answers.append({
            "title": item["title"],
            "file_name": item["file_name"],
            "section": item["section"],
            "rerank_score": item.get("rerank_score", 0.0),
            "content": item.get("expanded_content", item["content"]),
            "answer": answer_text
        })

    return answers


def print_generated_answers(answer_items: list[dict]):
    for i, item in enumerate(answer_items, start=1):
        print("=" * 80)
        print(f"[답변 {i}]")
        print(f"문서명       : {item['title']}")
        print(f"파일명       : {item['file_name']}")
        print(f"섹션         : {item['section']}")
        print(f"rerank_score : {item['rerank_score']:.4f}")
        print("-" * 80)
        print(item["answer"])
        print()


def answer_user_query(query: str, model: str = CHAT_MODEL) -> dict:
    query_type = classify_query(query)

    if query_type == "unsupported":
        return {
            "type": "unsupported",
            "answer": build_unsupported_response()
        }

    if query_type == "service_intro":
        intro_doc = load_service_intro_document(DATA_DIR)

        if not intro_doc:
            return {
                "type": "error",
                "answer": "서비스 소개 문서를 찾을 수 없습니다. 입지너구리_서비스소개.md 파일을 확인해주세요."
            }

        answer_text = generate_single_answer(query, intro_doc, model=model)
        return {
            "type": "service_intro",
            "results": [intro_doc],
            "answers": [{
                "title": intro_doc["title"],
                "file_name": intro_doc["file_name"],
                "section": intro_doc["section"],
                "rerank_score": 1.0,
                "content": intro_doc.get("expanded_content", intro_doc["content"]),
                "answer": answer_text
            }]
        }

    # 정책/대출 질문은 기존 검색 흐름 사용
    results = hybrid_search(query)
    answer_items = generate_answers_per_result(query, results, model=model)

    return {
        "type": "policy_or_loan",
        "results": results,
        "answers": answer_items
    }



# =========================
# CLI
# =========================
def print_search_results(results: list[dict], max_items: int = 5):
    for i, item in enumerate(results[:max_items], start=1):
        print("=" * 80)
        print(f"[{i}] title        : {item['title']}")
        print(f"    file         : {item['file_name']}")
        print(f"    section      : {item['section']}")
        print(f"    parent_h1    : {item.get('parent_h1', '')}")
        print(f"    rerank_score : {item.get('rerank_score', 0.0):.4f}")
        print(f"    하위섹션 전체 :")

        grouped_sections = item.get("grouped_sections", [])
        if not grouped_sections:
            print(item["content"][:500])
            if len(item["content"]) > 500:
                print("    ...")
            continue

        for section_item in grouped_sections:
            print(f"\n    [{section_item['section']}]")
            text = section_item["content"]
            print(text[:1000])
            if len(text) > 1000:
                print("    ...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["index", "search"])
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    if args.command == "index":
        index_markdown_files(DATA_DIR)

    elif args.command == "search":
        if not args.query:
            raise ValueError("search 명령에는 --query가 필요합니다.")

        response = answer_user_query(args.query, model=CHAT_MODEL)

        if response["type"] == "unsupported":
            print("\n[최종 답변]")
            print(response["answer"])

        elif response["type"] == "error":
            print("\n[오류]")
            print(response["answer"])

        else:
            print("\n[검색 결과]")
            if response.get("results"):
                print_search_results(response["results"])

            print("\n[최종 답변]")
            print_generated_answers(response["answers"])
        
        
