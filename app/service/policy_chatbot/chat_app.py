import logging
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set

import pymysql
from openai import OpenAI
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.database import get_connection
from app.service.policy_chatbot import data_indexing
from app.schema.policy_chatbot_schema import PolicyChatbotAskRequest

_OPENAI_CLIENT = OpenAI(api_key=settings.OPENAI_API_KEY)
_POLICY_CHATBOT_RERANKER = None
_POLICY_CHATBOT_RERANKER_LOCK = Lock()
_POLICY_CHATBOT_EMBEDDER = None
_POLICY_CHATBOT_EMBEDDER_LOCK = Lock()
_POLICY_CHATBOT_FAISS = None
_POLICY_CHATBOT_FAISS_ID_MAP = None
_POLICY_CHATBOT_FAISS_LOCK = Lock()
_POLICY_CHATBOT_BM25 = None
_POLICY_CHATBOT_BM25_DOCS = None
_POLICY_CHATBOT_BM25_DOC_IDX_BY_CHUNK_ID = None
_POLICY_CHATBOT_BM25_LOCK = Lock()

MAX_EXPANDED_CONTENT_CHARS = int(
    getattr(settings, "SEARCH_EXPANDED_CONTENT_MAX_CHARS", 2000)
)
MAX_LLM_CONTEXT_CHARS = int(
    getattr(settings, "SEARCH_LLM_CONTEXT_MAX_CHARS", 4200)
)

SEOUL_DISTRICTS = {
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구",
}
SEOUL_WIDE_REGION_KEYWORDS = ["서울시", "서울특별시", "서울", "전국"]
UNSELECTED_REGION_VALUES = {"", "미선택", "지역 미선택", "선택 안함", "선택안함", "전체", "전국"}


def now_ms() -> int:
    return int(time.perf_counter() * 1000)


def clip_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def initialize_policy_chatbot_resources() -> None:
    try:
        get_policy_chatbot_reranker()
        logging.info("[INIT] policy_chatbot resources initialized")
    except Exception as e:
        logging.exception("[INIT] policy_chatbot resource initialization failed: %s", e)


def get_openai_client() -> OpenAI:
    return _OPENAI_CLIENT


def get_policy_chatbot_embedder() -> "data_indexing.Embedder":
    global _POLICY_CHATBOT_EMBEDDER
    if _POLICY_CHATBOT_EMBEDDER is None:
        with _POLICY_CHATBOT_EMBEDDER_LOCK:
            if _POLICY_CHATBOT_EMBEDDER is None:
                _POLICY_CHATBOT_EMBEDDER = data_indexing.Embedder(
                    get_openai_client(),
                    settings.EMBED_MODEL,
                )
    return _POLICY_CHATBOT_EMBEDDER


def get_policy_chatbot_faiss():
    global _POLICY_CHATBOT_FAISS, _POLICY_CHATBOT_FAISS_ID_MAP
    if _POLICY_CHATBOT_FAISS is None or _POLICY_CHATBOT_FAISS_ID_MAP is None:
        with _POLICY_CHATBOT_FAISS_LOCK:
            if _POLICY_CHATBOT_FAISS is None or _POLICY_CHATBOT_FAISS_ID_MAP is None:
                _POLICY_CHATBOT_FAISS, _POLICY_CHATBOT_FAISS_ID_MAP = data_indexing.load_faiss_index(
                    settings.INDEX_DIR
                )
    return _POLICY_CHATBOT_FAISS, _POLICY_CHATBOT_FAISS_ID_MAP


def get_policy_chatbot_bm25():
    global _POLICY_CHATBOT_BM25, _POLICY_CHATBOT_BM25_DOCS, _POLICY_CHATBOT_BM25_DOC_IDX_BY_CHUNK_ID
    if (
        _POLICY_CHATBOT_BM25 is None
        or _POLICY_CHATBOT_BM25_DOCS is None
        or _POLICY_CHATBOT_BM25_DOC_IDX_BY_CHUNK_ID is None
    ):
        with _POLICY_CHATBOT_BM25_LOCK:
            if (
                _POLICY_CHATBOT_BM25 is None
                or _POLICY_CHATBOT_BM25_DOCS is None
                or _POLICY_CHATBOT_BM25_DOC_IDX_BY_CHUNK_ID is None
            ):
                _POLICY_CHATBOT_BM25, _POLICY_CHATBOT_BM25_DOCS = data_indexing.load_bm25_index(
                    settings.INDEX_DIR
                )
                _POLICY_CHATBOT_BM25_DOC_IDX_BY_CHUNK_ID = {
                    int(doc["chunk_id"]): idx
                    for idx, doc in enumerate(_POLICY_CHATBOT_BM25_DOCS)
                    if doc.get("chunk_id") is not None
                }
    return _POLICY_CHATBOT_BM25, _POLICY_CHATBOT_BM25_DOCS, _POLICY_CHATBOT_BM25_DOC_IDX_BY_CHUNK_ID


def get_policy_chatbot_reranker() -> "PolicyChatbotReranker":
    global _POLICY_CHATBOT_RERANKER

    if _POLICY_CHATBOT_RERANKER is None:
        with _POLICY_CHATBOT_RERANKER_LOCK:
            if _POLICY_CHATBOT_RERANKER is None:
                logging.info("[INIT] PolicyChatbot Reranker loading: %s", settings.RERANK_MODEL)
                _POLICY_CHATBOT_RERANKER = PolicyChatbotReranker(settings.RERANK_MODEL)
                logging.info("[INIT] PolicyChatbot Reranker loaded successfully")
    return _POLICY_CHATBOT_RERANKER


# =========================
# 공통 응답 빌더
# =========================
def build_response(
    response_type: str,
    user_query: str,
    answer: str,
    retrieved_documents: List[dict],
    retrieval_ms: int,
    llm_generation_ms: int,
    turn_latency_ms: int,
    session_title_suggestion: str,
) -> dict:
    return {
        "type": response_type,
        "answer": answer,
        "model": settings.CHAT_MODEL,
        "session_title_suggestion": session_title_suggestion,
        "retrieved_documents": retrieved_documents,
        "latency": {
            "retrieval_ms": retrieval_ms,
            "llm_generation_ms": llm_generation_ms,
            "turn_latency_ms": turn_latency_ms,
        },
        "chat_log_payload": {
            "user_query": user_query,
            "bot_response": answer,
            "model": settings.CHAT_MODEL,
            "turn_latency_ms": turn_latency_ms,
        },
        "rag_log_payload": {
            "retrieved_documents": retrieved_documents,
            "system_latency": {
                "retrieval_ms": retrieval_ms,
                "llm_generation_ms": llm_generation_ms,
            },
        },
    }


# =========================
# 서비스 소개 문서
# =========================
def find_service_intro_file(data_dir: str) -> Optional[Path]:
    data_path = Path(data_dir)
    for file_path in data_path.rglob("입지너구리_서비스소개.md"):
        return file_path
    return None


def load_service_intro_document(data_dir: str) -> Optional[dict]:
    file_path = find_service_intro_file(data_dir)
    if not file_path:
        return None

    title, chunks = data_indexing.build_chunks_from_markdown(file_path)

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
        "document_type": "service_intro",
        "source": file_path.name,
        "faiss_score": None,
        "bm25_score": None,
        "rerank_score": None,
    }


# =========================
# 분류 / 타이틀 / 질의 확장
# =========================
def classify_query(query: str, model: str = settings.CHAT_MODEL) -> str:
    q = (query or "").strip().lower()
    if not q:
        return "unsupported"

    service_keywords = [
        "입지너구리",
        "서비스",
        "기능",
        "사용법",
        "어떻게 써",
        "플랫폼",
    ]
    policy_keywords = [
        "정책",
        "지원",
        "지원금",
        "보조금",
        "대출",
        "정책자금",
        "보증",
        "금리",
        "한도",
        "상환",
        "창업",
        "소상공인",
    ]

    has_service = any(k in q for k in service_keywords)
    has_policy = any(k in q for k in policy_keywords)

    if has_policy:
        return "policy_or_loan"
    if has_service:
        return "service_intro"
    return "unsupported"


def suggest_session_title(user_query: str, answer: str = "", model: str = settings.CHAT_MODEL) -> str:
    query = (user_query or "").strip().replace("\n", " ")
    return query if query else "새 채팅"


def build_retrieval_query(user_query: str, user_profile: Optional[dict]) -> str:
    query = (user_query or "").strip()
    if not user_profile:
        return query

<<<<<<< HEAD
    parts = [query]

    industry = user_profile.get("industry")
    age = user_profile.get("age")
    has_business_registration = user_profile.get("has_business_registration")
    business_stage = user_profile.get("business_stage")
    region = _normalize_region(user_profile.get("region"))

    if industry:
        parts.append(f"업종:{industry}")
    if age is not None:
        parts.append(f"나이:{age}")
    if has_business_registration is not None:
        biz = "Y" if has_business_registration else "N"
        parts.append(f"사업자등록:{biz}")
    if business_stage:
        parts.append(f"사업여부:{business_stage}")
    if region:
        parts.append(f"지역:{region}")

    return " | ".join(parts)
=======
    if user_profile:
        industry = user_profile.get("industry")
        age = user_profile.get("age")
        has_business_registration = user_profile.get("has_business_registration")
        region = user_profile.get("region")
        startup_status = user_profile.get("startup_status")

        if industry:
            parts.append(f"업종: {industry}")
        if age is not None:
            parts.append(f"나이: {age}세")
            if age <= 39:
                parts.append("청년 관련 조건 확인")
            elif age >= 40:
                parts.append("중장년 관련 조건 확인")
        if has_business_registration is not None:
            parts.append(
                "사업자등록 있음" if has_business_registration else "사업자등록 없음"
            )
        if region:
            parts.append(f"지역: {region}")
        if startup_status:
            parts.append(f"창업 상태: {startup_status}")

    parts.append("지원대상, 신청자격, 나이 조건, 사업자등록 요건, 업종 요건, 지역 요건, 창업 단계(예비/운영/재창업) 요건, 신청방법, 신청기간 중심으로 검색")
    return "\n".join(parts)
>>>>>>> 352b0f45166f1e1e740c524bd936e828db48b6b9


# =========================
# 검색용 DB 조회
# =========================
def fetch_chunks_by_ids(chunk_ids: List[int]) -> Dict[int, Dict]:
    if not chunk_ids:
        return {}

    conn = None
    try:
        conn = get_connection()
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

    except Exception as e:
        logging.exception("[ERROR] fetch_chunks_by_ids 실패: %s", e)
        return {}

    finally:
        if conn:
            conn.close()


def fetch_chunks_by_parent_h1(document_id: int, parent_h1: str) -> List[Dict]:
    conn = None
    try:
        conn = get_connection()
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

    except Exception as e:
        logging.exception("[ERROR] fetch_chunks_by_parent_h1 실패: %s", e)
        return []

    finally:
        if conn:
            conn.close()


def expand_result_to_parent_context(item: Dict) -> Dict:
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
    expanded["expanded_content"] = clip_text(merged_content, MAX_EXPANDED_CONTENT_CHARS)
    expanded["grouped_sections"] = merged_sections
    return expanded


# =========================
# Metadata pre-filter
# =========================
def _age_keywords(age: Optional[int]) -> List[str]:
    if age is None:
        return []
    if age <= 39:
        return ["청년", "39세 이하", "만 39세", "청년창업", "청년전용"]
    return ["중장년", "40세 이상", "만 40세", "신중년"]


def _business_stage_keywords(stage: Optional[str]) -> List[str]:
    mapping = {
        "예비창업": ["예비창업", "창업예정", "창업 준비", "미창업"],
        "창업": ["기창업", "개인사업자", "법인사업자", "사업자"],
        "재창업": ["재창업", "폐업", "재도전", "재기"],
    }
    return mapping.get(stage, [])


def _normalize_region(region: Optional[str]) -> str:
    region = (region or "").strip()
    if region in UNSELECTED_REGION_VALUES:
        return ""
    return region


def _has_region_selection(user_profile: Optional[dict]) -> bool:
    if not user_profile:
        return False
    raw_region = (user_profile.get("region") or "").strip()
    return raw_region not in UNSELECTED_REGION_VALUES


def _build_search_blob(item: Dict) -> str:
    return " ".join([
        str(item.get("title") or ""),
        str(item.get("section") or ""),
        str(item.get("parent_h1") or ""),
        str(item.get("parent_h2") or ""),
        str(item.get("content") or ""),
        str(item.get("file_name") or ""),
        str(item.get("source_path") or ""),
    ])


def _has_other_district(blob: str, region: Optional[str] = None) -> bool:
    region = _normalize_region(region)
    other_districts = SEOUL_DISTRICTS - {region}
    return any(other_region in blob for other_region in other_districts)


def _is_wide_region_blob(blob: str) -> bool:
    return not _has_other_district(blob) and any(
        keyword in blob for keyword in SEOUL_WIDE_REGION_KEYWORDS
    )


def _is_selected_region_blob(blob: str, region: str) -> bool:
    return not _has_other_district(blob, region) and region in blob


def _is_allowed_for_region(blob: str, region: Optional[str]) -> bool:
    region = _normalize_region(region)
    if region in blob:
        return _is_selected_region_blob(blob, region)

    return _is_wide_region_blob(blob)


def _profile_match_count(item: Dict, user_profile: Optional[dict]) -> int:
    if not user_profile:
        return 0

    blob = _build_search_blob(item)

    score = 0
    industry = user_profile.get("industry")
    region = user_profile.get("region")
    age = user_profile.get("age")
    has_business_registration = user_profile.get("has_business_registration")
    business_stage = user_profile.get("business_stage")

    if industry and industry in blob:
        score += 1
    if region and region in blob:
        score += 1
    if age is not None and any(k in blob for k in _age_keywords(age)):
        score += 1
    if business_stage and any(k in blob for k in _business_stage_keywords(business_stage)):
        score += 1

    if has_business_registration is True and any(
        k in blob for k in ["사업자등록", "개인사업자", "법인사업자", "사업자"]
    ):
        score += 1
    if has_business_registration is False and any(
        k in blob for k in ["예비창업", "사업자등록 전", "미등록", "창업예정"]
    ):
        score += 1

    return score


def apply_profile_prefilter(results: List[Dict], user_profile: Optional[dict]) -> List[Dict]:
    if not results or not user_profile:
        return results

    scored = []
    for item in results:
        match_count = _profile_match_count(item, user_profile)
        new_item = dict(item)
        new_item["profile_match_count"] = match_count
        scored.append(new_item)

    filtered = [x for x in scored if x["profile_match_count"] > 0]
    target = filtered if filtered else scored
    target.sort(key=lambda x: x["profile_match_count"], reverse=True)
    return target


def _build_profile_filter_keywords(user_profile: Optional[dict]) -> List[str]:
    if not user_profile:
        return []

    keywords: List[str] = []
    industry = user_profile.get("industry")
    age = user_profile.get("age")
    has_business_registration = user_profile.get("has_business_registration")
    business_stage = user_profile.get("business_stage")

    if industry:
        keywords.append(str(industry))
    if age is not None:
        keywords.extend(_age_keywords(age))
    if business_stage:
        keywords.extend(_business_stage_keywords(business_stage))
    if has_business_registration is True:
        keywords.extend(["사업자등록", "개인사업자", "법인사업자"])
    elif has_business_registration is False:
        keywords.extend(["예비창업", "사업자등록 전", "미등록", "창업예정"])

    # 길이 2 이상 키워드만 사용하고, 중복 제거
    deduped = []
    seen = set()
    for k in keywords:
        k = (k or "").strip()
        if len(k) < 2 or k in seen:
            continue
        seen.add(k)
        deduped.append(k)
    return deduped


def fetch_region_allowed_chunk_ids(user_profile: Optional[dict], limit: int = 3000) -> Optional[Set[int]]:
    if not user_profile:
        return None

    region = _normalize_region(user_profile.get("region"))
    has_region_selection = _has_region_selection(user_profile)

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            keywords = [*SEOUL_WIDE_REGION_KEYWORDS]
            if region:
                keywords.insert(0, region)

            where_parts = []
            params: List[str] = []

            for kw in keywords:
                like_kw = f"%{kw}%"
                where_parts.append(
                    "("
                    "d.TITLE LIKE %s OR "
                    "d.FILE_NAME LIKE %s OR "
                    "d.SOURCE_PATH LIKE %s OR "
                    "c.SECTION LIKE %s OR "
                    "c.PARENT_H1 LIKE %s OR "
                    "c.PARENT_H2 LIKE %s OR "
                    "c.CONTENT LIKE %s"
                    ")"
                )
                params.extend([like_kw, like_kw, like_kw, like_kw, like_kw, like_kw, like_kw])

            sql = f"""
                SELECT
                    c.CHUNKS_ID AS chunk_id,
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
                WHERE {" OR ".join(where_parts)}
                ORDER BY c.CHUNKS_ID ASC
                LIMIT %s
            """
            params.append(int(limit))
            cursor.execute(sql, params)
            rows = cursor.fetchall()

            selected_region_ids: Set[int] = set()
            wide_region_ids: Set[int] = set()

            for row in rows:
                if row.get("chunk_id") is None:
                    continue

                chunk_id = int(row["chunk_id"])
                blob = _build_search_blob(row)

                if region and _is_selected_region_blob(blob, region):
                    selected_region_ids.add(chunk_id)
                elif _is_wide_region_blob(blob):
                    wide_region_ids.add(chunk_id)

            if has_region_selection and selected_region_ids:
                return selected_region_ids

            return wide_region_ids

    except Exception as e:
        logging.exception("[ERROR] fetch_region_allowed_chunk_ids 실패: %s", e)
        return set()

    finally:
        if conn:
            conn.close()


def fetch_candidate_chunk_ids(user_profile: Optional[dict], limit: int = 3000) -> Optional[Set[int]]:
    keywords = _build_profile_filter_keywords(user_profile)
    if not keywords:
        return None

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            where_parts = []
            params: List[str] = []
            for kw in keywords:
                like_kw = f"%{kw}%"
                where_parts.append(
                    "("
                    "d.TITLE LIKE %s OR "
                    "c.SECTION LIKE %s OR "
                    "c.PARENT_H1 LIKE %s OR "
                    "c.PARENT_H2 LIKE %s OR "
                    "c.CONTENT LIKE %s"
                    ")"
                )
                params.extend([like_kw, like_kw, like_kw, like_kw, like_kw])

            sql = f"""
                SELECT c.CHUNKS_ID AS chunk_id
                FROM CHATBOT_DATA_CHUNKS c
                JOIN CHATBOT_DOCUMENTS d
                  ON c.DOCUMENT_ID = d.DOCUMENT_ID
                WHERE {" OR ".join(where_parts)}
                ORDER BY c.CHUNKS_ID ASC
                LIMIT %s
            """
            params.append(int(limit))
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            candidate_ids = {int(row["chunk_id"]) for row in rows if row.get("chunk_id") is not None}
            if not candidate_ids:
                return None
            return candidate_ids

    except Exception as e:
        logging.exception("[ERROR] fetch_candidate_chunk_ids 실패: %s", e)
        return None

    finally:
        if conn:
            conn.close()


# =========================
# Reranker
# =========================
class PolicyChatbotReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(
            model_name,
            max_length=512,
            device="cpu",
        )

    def rerank(self, query: str, candidates: List[dict], top_k: int = 10):
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
# Search
# =========================
def semantic_search(query: str, top_k: int = 10, allowed_chunk_ids: Optional[Set[int]] = None) -> List[Dict]:
    try:
        if allowed_chunk_ids is not None and not allowed_chunk_ids:
            return []

        embedder = get_policy_chatbot_embedder()
        index, id_map = get_policy_chatbot_faiss()

        qvec = embedder.embed_query(query)
        qvec = data_indexing.normalize(qvec)

        search_k = top_k
        if allowed_chunk_ids is not None:
            search_k = len(id_map)

        scores, indices = index.search(qvec, search_k)

        candidate_ids = []
        score_map = {}

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = id_map[idx]
            if allowed_chunk_ids is not None and chunk_id not in allowed_chunk_ids:
                continue
            if chunk_id in score_map:
                continue
            candidate_ids.append(chunk_id)
            score_map[chunk_id] = float(score)
            if len(candidate_ids) >= top_k:
                break

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
                "source": chunk["file_name"],
                "faiss_score": score_map[chunk_id],
            })

        return results

    except Exception as e:
        logging.exception("[ERROR] semantic_search 실패: %s", e)
        return []


def bm25_search(query: str, top_k: int = 10, allowed_chunk_ids: Optional[Set[int]] = None) -> List[Dict]:
    try:
        if allowed_chunk_ids is not None and not allowed_chunk_ids:
            return []

        bm25, docs, doc_idx_by_chunk_id = get_policy_chatbot_bm25()

        query_tokens = data_indexing.bm25_tokenize(query)
        if allowed_chunk_ids is not None:
            allowed_doc_indices = [
                doc_idx_by_chunk_id[chunk_id]
                for chunk_id in allowed_chunk_ids
                if chunk_id in doc_idx_by_chunk_id
            ]
            if not allowed_doc_indices:
                return []

            batch_scores = bm25.get_batch_scores(query_tokens, allowed_doc_indices)
            ranked = sorted(
                (
                    (docs[doc_idx], score)
                    for doc_idx, score in zip(allowed_doc_indices, batch_scores)
                ),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
        else:
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
                "source": doc["file_name"],
                "bm25_score": float(score),
            })

        return results

    except Exception as e:
        logging.exception("[ERROR] bm25_search 실패: %s", e)
        return []


def rrf_fusion(semantic_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
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
            "source": item.get("source"),
            "faiss_score": item.get("faiss_score"),
            "bm25_score": None,
            "rrf_score": 0.0,
        })
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
            "source": item.get("source"),
            "faiss_score": None,
            "bm25_score": item.get("bm25_score"),
            "rrf_score": 0.0,
        })
        if merged[chunk_id].get("bm25_score") is None:
            merged[chunk_id]["bm25_score"] = item.get("bm25_score")
        merged[chunk_id]["rrf_score"] += 1.0 / (k + rank)

    fused = list(merged.values())
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


def hybrid_search(query: str, user_profile: Optional[dict] = None) -> List[Dict]:
    try:
        prefilter_limit = int(getattr(settings, "SEARCH_PREFILTER_MAX_CANDIDATES", 3000))
        profile_candidate_ids = fetch_candidate_chunk_ids(user_profile, limit=prefilter_limit)
        region_allowed_ids = fetch_region_allowed_chunk_ids(user_profile, limit=prefilter_limit)

        allowed_chunk_ids = profile_candidate_ids
        if region_allowed_ids is not None:
            allowed_chunk_ids = (
                region_allowed_ids
                if allowed_chunk_ids is None
                else allowed_chunk_ids & region_allowed_ids
            )

        semantic_results = semantic_search(
            query,
            top_k=settings.SEARCH_SEMANTIC_TOP_K,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        bm25_results = bm25_search(
            query,
            top_k=settings.SEARCH_BM25_TOP_K,
            allowed_chunk_ids=allowed_chunk_ids,
        )

        semantic_results = apply_profile_prefilter(semantic_results, user_profile)
        bm25_results = apply_profile_prefilter(bm25_results, user_profile)

        if not semantic_results and not bm25_results:
            return []

        fused = rrf_fusion(semantic_results, bm25_results)
        fused = fused[:settings.SEARCH_FUSED_TOP_K]

        reranker = get_policy_chatbot_reranker()
        reranked = reranker.rerank(query, fused, top_k=settings.SEARCH_RERANK_TOP_K)

        expanded_results = [expand_result_to_parent_context(item) for item in reranked]
        return expanded_results

    except Exception as e:
        logging.exception("[ERROR] hybrid_search 실패: %s", e)
        return []


# =========================
# Prompt / Answer
# =========================
def build_user_context_text(user_profile: Optional[dict]) -> str:
    if not user_profile:
        return "사용자 추가 정보 없음"

    lines = []
    if user_profile.get("industry"):
        lines.append(f"- 업종: {user_profile['industry']}")
    if user_profile.get("age") is not None:
        lines.append(f"- 나이: {user_profile['age']}세")
    if user_profile.get("has_business_registration") is not None:
        lines.append(
            f"- 사업자등록 여부: {'있음' if user_profile['has_business_registration'] else '없음'}"
        )
<<<<<<< HEAD
    if user_profile.get("business_stage"):
        lines.append(f"- 사업여부: {user_profile['business_stage']}")
    region = _normalize_region(user_profile.get("region"))
    if region:
        lines.append(f"- 지역: {region}")
=======
    if user_profile.get("region"):
        lines.append(f"- 지역: {user_profile['region']}")
    if user_profile.get("startup_status"):
        lines.append(f"- 창업 상태: {user_profile['startup_status']}")
>>>>>>> 352b0f45166f1e1e740c524bd936e828db48b6b9

    return "\n".join(lines) if lines else "사용자 추가 정보 없음"


def build_context_from_results(results: List[Dict], max_items: int = 5) -> str:
    selected = results[:max_items]
    parts = []
    current_len = 0

    for i, item in enumerate(selected, start=1):
        if current_len >= MAX_LLM_CONTEXT_CHARS:
            break

        content_for_answer = clip_text(
            item.get("expanded_content") or item.get("content", ""),
            MAX_EXPANDED_CONTENT_CHARS,
        )

        part = (
            f"[문서 {i}]\n"
            f"문서명: {item['title']}\n"
            f"파일명: {item['file_name']}\n"
            f"섹션: {item['section']}\n"
            f"faiss_score: {item.get('faiss_score')}\n"
            f"bm25_score: {item.get('bm25_score')}\n"
            f"rerank_score: {item.get('rerank_score')}\n"
            f"내용:\n{content_for_answer}\n"
        )

        remaining = MAX_LLM_CONTEXT_CHARS - current_len
        if remaining <= 0:
            break
        part = clip_text(part, remaining)
        parts.append(part)
        current_len += len(part) + 2

    return "\n\n".join(parts)


def build_service_intro_prompt(user_query: str, item: dict) -> str:
    content_for_answer = clip_text(
        item.get("expanded_content", item["content"]),
        MAX_LLM_CONTEXT_CHARS,
    )

    return f"""
    역할: 너는 ‘입지너구리’ 서비스 소개 챗봇이다. 사용자가 ‘입지너구리’ 서비스에 대해 궁금해하는 질문에 답변한다.

    규칙:
    - 반드시 [문서 내용]에 있는 정보만 사용해 답변한다.
    - 문서에 없는 내용은 추측하거나 생성하지 않는다.
    - 사용자 질문과 관련된 내용만 간결하게 설명한다.
    - 말투는 항상 친절한 한국어로 작성한다. ("~이에요", "~가 있어요", "~할 수 있어요")
    - 큰 항목은 **굵게** 표시한다.
    - 필요할 때만 "---"로 구분한다.
    - 중요한 내용에는 이모티콘을 적절히 사용할 수 있다.
    - 답변 마지막에 추가 제안, 선택지, 다음 단계 안내는 쓰지 않는다.
    - "원하시면 ~", "추가로 ~ 도와드릴 수 있어요" 같은 문장은 금지한다.

    [사용자 질문]
    {user_query}

    [문서 내용]
    {content_for_answer}
    """.strip()


def build_multi_result_prompt(user_query: str, user_profile: Optional[dict], results: List[Dict]) -> str:
    context = build_context_from_results(
        results,
        max_items=min(settings.SEARCH_RESPONSE_TOP_K, len(results))
    )
    profile_text = build_user_context_text(user_profile)

    return f"""
    역할: 너는 한국의 정부지원정책, 창업지원, 정책자금, 대출 정보를 소상공인에게 안내하는 한국어 챗봇이다.

    규칙:
    - 반드시 [검색 결과 문서]에 있는 내용만 사용해 답변한다.
    - 문서에 없는 내용은 추측하거나 생성하지 않는다.
    - [사용자 추가 정보]가 있으면 문서의 지원대상, 신청조건, 자격요건과 비교해 적합 여부를 설명한다.
    - "문서에는 ~"과 같이 문서를 근거로 한다는 문장은 사용하지 않는다.
    - 사용자와 더 잘 맞는 정책/상품을 우선 소개한다.
    - 사용자에게 적합하지 않는 정책/상품은 제외한다.
    - 적합한 이유는 문서의 내용을 근거로 짧고 분명하게 설명한다.
    - 문서에 없는 조건은 판단하지 말고 "추가 확인이 필요해요"라고 답한다.
    - 말투는 친절한 한국어로 작성한다. ("~이에요", "~가 있어요", "~할 수 있어요")
    - 정책명/상품명은 **굵게** 표시한다.
    - 필요할 때만 "---"로 구분한다.
    - 중요한 내용에는 이모티콘을 적절히 사용할 수 있다.
    - URL은 문서에 실제 url이 있는 경우에만 마지막에 "🔗 자세히 보기: URL" 형식으로 포함한다.
    - 값이 "없음", "데이터 없음", "null", "None", 공백, 빈 문자열처럼 실질적으로 값이 없는 경우에는 출력하지 않는다.
    - 답변 마지막에 추가 질문 유도, 다음 단계 제안은 쓰지 않는다.
    - "원하시면 ~", "추가로 ~ 도와드릴 수 있어요" 같은 문장은 금지한다.

    [사용자 질문]
    {user_query}

    [사용자 추가 정보]
    {profile_text}

    [검색 결과 문서]
    {context}
    """.strip()


def generate_service_answer(query: str, item: dict, model: str) -> str:
    client = get_openai_client()
    prompt = build_service_intro_prompt(query, item)
    response = client.responses.create(model=model, input=prompt)
    return response.output_text.strip()


def generate_combined_answer(query: str, user_profile: Optional[dict], results: List[Dict], model: str) -> str:
    if not results:
        return "관련 문서를 찾지 못해서 답변을 드리기 어려워요. 질문을 조금 더 구체적으로 적어주세요."

    client = get_openai_client()
    prompt = build_multi_result_prompt(query, user_profile, results)
    response = client.responses.create(model=model, input=prompt)
    return response.output_text.strip()


def build_retrieved_documents_for_log(results: List[Dict]) -> List[Dict]:
    docs = []

    for item in results[:settings.SEARCH_RESPONSE_TOP_K]:
        docs.append({
            "source": item.get("source") or item.get("file_name") or "unknown",
            "chunk_text": item.get("content", "")[:1000],
            "faiss_score": item.get("faiss_score"),
            "bm25_score": item.get("bm25_score"),
            "rerank_score": item.get("rerank_score"),
        })

    return docs


# =========================
# Main
# =========================
def answer_policy_chatbot_query(request: PolicyChatbotAskRequest) -> dict:
    turn_start = now_ms()
    retrieval_ms = 0
    llm_generation_ms = 0

    user_query = request.user_query.strip()
    user_profile = request.user_profile.model_dump() if request.user_profile else None

    try:
        if not user_query:
            return build_response(
                response_type="error",
                user_query="",
                answer="질문이 비어 있습니다. 내용을 입력해주세요.",
                retrieved_documents=[],
                retrieval_ms=0,
                llm_generation_ms=0,
                turn_latency_ms=0,
                session_title_suggestion="새 채팅",
            )

        query_type = classify_query(user_query, model=settings.CHAT_MODEL)

        if query_type == "unsupported":
            answer = (
                "현재 이 챗봇은 정부지원정책, 창업지원, 보조금, 정책자금, 대출, "
                "그리고 서비스 소개 관련 질문만 답변할 수 있어요."
            )
            title = suggest_session_title(user_query, answer)
            turn_latency_ms = now_ms() - turn_start

            return build_response(
                response_type="unsupported",
                user_query=user_query,
                answer=answer,
                retrieved_documents=[],
                retrieval_ms=0,
                llm_generation_ms=0,
                turn_latency_ms=turn_latency_ms,
                session_title_suggestion=title,
            )

        if query_type == "service_intro":
            intro_doc = load_service_intro_document(settings.DATA_DIR)

            if not intro_doc:
                raise FileNotFoundError("입지너구리_서비스소개.md 파일을 찾을 수 없습니다.")

            llm_start = now_ms()
            answer = generate_service_answer(user_query, intro_doc, model=settings.CHAT_MODEL)
            llm_generation_ms = now_ms() - llm_start
            turn_latency_ms = now_ms() - turn_start
            title = suggest_session_title(user_query, answer)
            retrieved_documents = build_retrieved_documents_for_log([intro_doc])

            return build_response(
                response_type="service_intro",
                user_query=user_query,
                answer=answer,
                retrieved_documents=retrieved_documents,
                retrieval_ms=0,
                llm_generation_ms=llm_generation_ms,
                turn_latency_ms=turn_latency_ms,
                session_title_suggestion=title,
            )

        retrieval_query = build_retrieval_query(user_query, user_profile)

        retrieval_start = now_ms()
        results = hybrid_search(retrieval_query, user_profile=user_profile)
        retrieval_ms = now_ms() - retrieval_start

        if not results:
            region = _normalize_region(user_profile.get("region") if user_profile else None)
            if region:
                answer = (
                    f"{region} 또는 서울시/전국 단위로 적용 가능한 정책·대출 문서를 찾지 못했어요. "
                    "현재 등록된 문서 기준으로는 해당 지역에 맞는 결과가 없습니다."
                )
            else:
                answer = "관련된 정책 또는 대출 문서를 찾지 못했어요. 질문을 조금 더 구체적으로 적어주시면 도움이 될 수 있어요."
            title = suggest_session_title(user_query, answer)
            turn_latency_ms = now_ms() - turn_start

            return build_response(
                response_type="policy_or_loan",
                user_query=user_query,
                answer=answer,
                retrieved_documents=[],
                retrieval_ms=retrieval_ms,
                llm_generation_ms=0,
                turn_latency_ms=turn_latency_ms,
                session_title_suggestion=title,
            )

        llm_start = now_ms()
        answer = generate_combined_answer(
            query=user_query,
            user_profile=user_profile,
            results=results,
            model=settings.CHAT_MODEL,
        )
        llm_generation_ms = now_ms() - llm_start
        turn_latency_ms = now_ms() - turn_start
        title = suggest_session_title(user_query, answer)
        retrieved_documents = build_retrieved_documents_for_log(results)

        return build_response(
            response_type="policy_or_loan",
            user_query=user_query,
            answer=answer,
            retrieved_documents=retrieved_documents,
            retrieval_ms=retrieval_ms,
            llm_generation_ms=llm_generation_ms,
            turn_latency_ms=turn_latency_ms,
            session_title_suggestion=title,
        )

    except FileNotFoundError as e:
        logging.exception("[ERROR] 파일 없음: %s", e)
        turn_latency_ms = now_ms() - turn_start
        return build_response(
            response_type="error",
            user_query=user_query,
            answer=f"검색에 필요한 인덱스 파일이 없습니다. 먼저 인덱싱을 실행해주세요. ({e})",
            retrieved_documents=[],
            retrieval_ms=0,
            llm_generation_ms=0,
            turn_latency_ms=turn_latency_ms,
            session_title_suggestion=user_query[:15] if user_query else "새 채팅",
        )

    except pymysql.MySQLError as e:
        logging.exception("[ERROR] DB 오류: %s", e)
        turn_latency_ms = now_ms() - turn_start
        return build_response(
            response_type="error",
            user_query=user_query,
            answer="검색용 데이터베이스 처리 중 오류가 발생했어요. DB 연결과 chunk 테이블 상태를 확인해주세요.",
            retrieved_documents=[],
            retrieval_ms=0,
            llm_generation_ms=0,
            turn_latency_ms=turn_latency_ms,
            session_title_suggestion=user_query[:15] if user_query else "새 채팅",
        )

    except Exception as e:
        logging.exception("[ERROR] answer_policy_chatbot_query 실패: %s", e)
        turn_latency_ms = now_ms() - turn_start
        return build_response(
            response_type="error",
            user_query=user_query,
            answer="질문 처리 중 예상치 못한 오류가 발생했어요. 잠시 후 다시 시도해주세요.",
            retrieved_documents=[],
            retrieval_ms=0,
            llm_generation_ms=0,
            turn_latency_ms=turn_latency_ms,
            session_title_suggestion=user_query[:15] if user_query else "새 채팅",
        )
