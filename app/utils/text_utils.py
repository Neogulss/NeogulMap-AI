import json
import math
import re


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def convert_markdown_links(text: str) -> str:
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1: \2", text)


def strip_code_fences(text: str) -> str:
    return re.sub(r"```.*?```", "[CODE_BLOCK]", text, flags=re.DOTALL)


def preprocess_markdown(text: str) -> str:
    text = convert_markdown_links(text)
    text = strip_code_fences(text)
    text = normalize_whitespace(text)
    return text


def approximate_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 3))


def safe_json_loads(text: str) -> dict:
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("JSON 객체를 찾을 수 없습니다.")

    return json.loads(text[start:end + 1])