import json
import os
import pickle
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pymysql
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

try:
    from app.utils.text_utils import approximate_token_count, preprocess_markdown
except ModuleNotFoundError:
    def _normalize_whitespace(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _convert_markdown_links(text: str) -> str:
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1: \2", text)

    def _strip_code_fences(text: str) -> str:
        return re.sub(r"```.*?```", "[CODE_BLOCK]", text, flags=re.DOTALL)

    def preprocess_markdown(text: str) -> str:
        text = _convert_markdown_links(text)
        text = _strip_code_fences(text)
        text = _normalize_whitespace(text)
        return text

    def approximate_token_count(text: str) -> int:
        return max(1, math.ceil(len(text) / 3))

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} 환경변수가 필요합니다.")
    return value


def _get_mysql_connection():
    return pymysql.connect(
        host=_require_env("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=_require_env("MYSQL_USER"),
        password=_require_env("MYSQL_PASSWORD"),
        database=_require_env("MYSQL_DB"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def extract_title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        m = re.match(r"^#\s+(.*)$", line.strip())
        if m:
            return m.group(1).strip()
    return fallback


def header_to_title(header: str, fallback: str) -> str:
    if not header:
        return fallback
    return re.sub(r"^#+\s*", "", header).strip() or fallback


def split_by_markdown_headers(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    sections = []
    current_h1 = None
    current_h2 = None
    current_header = "문서 시작"
    current_content = []

    for line in lines:
        line = line.rstrip()
        if re.match(r"^#\s+", line) or re.match(r"^##\s+", line):
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    sections.append({"h1": current_h1, "h2": current_h2, "header": current_header, "content": content})
            if re.match(r"^#\s+", line):
                current_h1 = line.strip()
                current_h2 = None
            else:
                current_h2 = line.strip()
            current_header = line.strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            sections.append({"h1": current_h1, "h2": current_h2, "header": current_header, "content": content})
    return sections


def split_long_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks_from_markdown(file_path: Path, max_chars: int = 1200, overlap: int = 150) -> Tuple[str, List[Dict]]:
    raw_text = preprocess_markdown(file_path.read_text(encoding="utf-8"))
    file_name = file_path.name
    title = extract_title_from_markdown(raw_text, fallback=file_name)
    sections = split_by_markdown_headers(raw_text)
    chunks, chunk_index = [], 0
    for section in sections:
        h1 = section["h1"] or f"# {title}"
        section_title = header_to_title(h1, fallback=title)
        h2 = section["h2"]
        header = section["header"]
        content = section["content"]
        combined = (
            f"문서명: {section_title}\n파일명: {file_name}\n상위섹션: {h1}\n"
            f"하위섹션: {h2 or '없음'}\n현재섹션: {header}\n내용:\n{content}"
        )
        for chunk_text in split_long_text(combined, max_chars=max_chars, overlap=overlap):
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "file_name": file_name,
                    "title": section_title,
                    "section": header,
                    "parent_h1": h1,
                    "parent_h2": h2,
                    "content": chunk_text,
                    "token_count": approximate_token_count(chunk_text),
                }
            )
            chunk_index += 1
    return title, chunks


def bm25_tokenize(text: str) -> List[str]:
    return re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def sanitize_text_for_embedding(text: str) -> str:
    # Remove problematic control/surrogate characters that can break JSON serialization.
    text = (text or "").replace("\x00", " ")
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return text


class Embedder:
    def __init__(self, openai_client, model_name: str):
        self.client = openai_client
        self.model_name = model_name

    def embed_query(self, text: str):
        clean_text = sanitize_text_for_embedding(text)
        response = self.client.embeddings.create(model=self.model_name, input=[clean_text])
        return np.array([response.data[0].embedding], dtype="float32")

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [sanitize_text_for_embedding(t) for t in texts[i : i + batch_size]]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            for item in response.data:
                all_embeddings.append(item.embedding)
        return np.array(all_embeddings, dtype="float32")


def load_faiss_index(index_dir: str) -> Tuple[faiss.Index, List[int]]:
    faiss_path = os.path.join(index_dir, "faiss.index")
    id_map_path = os.path.join(index_dir, "id_map.json")
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS 인덱스 파일이 없습니다: {faiss_path}")
    if not os.path.exists(id_map_path):
        raise FileNotFoundError(f"id_map 파일이 없습니다: {id_map_path}")
    index = faiss.read_index(faiss_path)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return index, id_map


def save_faiss_index(index_dir: str, vectors: np.ndarray, id_map: List[int]) -> None:
    os.makedirs(index_dir, exist_ok=True)
    faiss_path = os.path.join(index_dir, "faiss.index")
    id_map_path = os.path.join(index_dir, "id_map.json")

    vectors = normalize(vectors.astype("float32"))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, faiss_path)

    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)


def load_bm25_index(index_dir: str):
    bm25_path = os.path.join(index_dir, "bm25.pkl")
    if not os.path.exists(bm25_path):
        raise FileNotFoundError(f"BM25 인덱스 파일이 없습니다: {bm25_path}")
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["docs"]


def save_bm25_index(index_dir: str, docs: List[Dict]) -> None:
    os.makedirs(index_dir, exist_ok=True)
    bm25 = BM25Okapi([bm25_tokenize(d.get("content", "")) for d in docs])
    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)


def build_docs_from_markdown_dir(data_dir: str, max_chars: int = 1200, overlap: int = 150) -> List[Dict]:
    md_files = sorted(Path(data_dir).rglob("*.md"))
    docs: List[Dict] = []

    for file_path in md_files:
        _, chunks = build_chunks_from_markdown(file_path, max_chars=max_chars, overlap=overlap)
        for chunk in chunks:
            docs.append(
                {
                    "chunk_id": None,
                    "document_id": None,
                    "chunk_index": chunk["chunk_index"],
                    "title": chunk["title"],
                    "file_name": chunk["file_name"],
                    # One markdown file can contain multiple top-level policies (# ...).
                    # Split DB documents by parent_h1 so each policy name is preserved.
                    "source_path": f"{file_path}::{chunk.get('parent_h1') or chunk['title']}",
                    "section": chunk["section"],
                    "parent_h1": chunk["parent_h1"],
                    "parent_h2": chunk["parent_h2"],
                    "content": chunk["content"],
                    "source": chunk["file_name"],
                }
            )

    return docs


def save_docs_to_db(docs: List[Dict]) -> Dict[str, int]:
    """
    Save chunks into CHATBOT_DOCUMENTS / CHATBOT_DATA_CHUNKS.
    Existing rows with same (FILE_NAME, SOURCE_PATH) are replaced.
    """
    conn = _get_mysql_connection()
    inserted_documents = 0
    inserted_chunks = 0

    try:
        with conn.cursor() as cursor:
            # Process grouped by source file to keep document/chunk relation clear.
            grouped: Dict[Tuple[str, str], List[Dict]] = {}
            for doc in docs:
                key = (doc["file_name"], doc["source_path"])
                grouped.setdefault(key, []).append(doc)

            for (file_name, source_path), group_docs in grouped.items():
                title = group_docs[0]["title"]

                # Replace old document+chunks for same source file.
                cursor.execute(
                    """
                    DELETE FROM CHATBOT_DOCUMENTS
                    WHERE FILE_NAME = %s AND SOURCE_PATH = %s
                    """,
                    (file_name, source_path),
                )

                cursor.execute(
                    """
                    INSERT INTO CHATBOT_DOCUMENTS (FILE_NAME, TITLE, SOURCE_PATH)
                    VALUES (%s, %s, %s)
                    """,
                    (file_name, title, source_path),
                )
                document_id = cursor.lastrowid
                inserted_documents += 1

                # Keep stable order for reproducible IDs/index alignment.
                group_docs.sort(key=lambda x: x["chunk_index"])

                for doc in group_docs:
                    cursor.execute(
                        """
                        INSERT INTO CHATBOT_DATA_CHUNKS
                        (DOCUMENT_ID, CHUNK_INDEX, SECTION, PARENT_H1, PARENT_H2, CONTENT, TOKEN_COUNT)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            document_id,
                            doc["chunk_index"],
                            doc["section"],
                            doc["parent_h1"],
                            doc["parent_h2"],
                            doc["content"],
                            approximate_token_count(doc["content"]),
                        ),
                    )
                    doc["document_id"] = document_id
                    doc["chunk_id"] = cursor.lastrowid
                    inserted_chunks += 1

        conn.commit()
        return {"documents": inserted_documents, "chunks": inserted_chunks}

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def assign_local_ids_for_non_db_mode(docs: List[Dict]) -> None:
    for idx, doc in enumerate(docs):
        doc["chunk_id"] = idx
        doc["document_id"] = idx


def build_and_save_indexes(data_dir: str, index_dir: str, embed_model: str, save_to_db: bool = True) -> Dict[str, int]:
    docs = build_docs_from_markdown_dir(data_dir)
    if not docs:
        raise FileNotFoundError(f"Markdown 파일이 없습니다: {data_dir}")

    if save_to_db:
        db_stats = save_docs_to_db(docs)
    else:
        assign_local_ids_for_non_db_mode(docs)
        db_stats = {"documents": len({(d["file_name"], d["source_path"]) for d in docs}), "chunks": len(docs)}

    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    embedder = Embedder(client, embed_model)
    vectors = embedder.embed_texts([doc["content"] for doc in docs])

    save_faiss_index(index_dir=index_dir, vectors=vectors, id_map=[doc["chunk_id"] for doc in docs])
    save_bm25_index(index_dir=index_dir, docs=docs)

    return db_stats


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = os.getenv("DATA_DIR")
    index_dir = os.getenv("INDEX_DIR")
    embed_model = os.getenv("EMBED_MODEL")
    save_to_db = os.getenv("SAVE_TO_DB", "true").lower() in {"1", "true", "y", "yes"}

    stats = build_and_save_indexes(
        data_dir=data_dir,
        index_dir=index_dir,
        embed_model=embed_model,
        save_to_db=save_to_db,
    )
    print(f"[DONE] documents={stats['documents']}, chunks={stats['chunks']}, save_to_db={save_to_db}")
