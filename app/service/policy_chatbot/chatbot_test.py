import sys
import types
import time
from pathlib import Path

import streamlit as st


def _bootstrap_local_app_package() -> None:
    """
    Ensure local project package 'app' is importable even if a site-packages 'app' exists.
    """
    repo_root = Path(__file__).resolve().parents[3]
    app_root = repo_root / "app"

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))

    # Force local namespace package to win over site-packages package named "app".
    app_module = types.ModuleType("app")
    app_module.__path__ = [str(app_root)]
    sys.modules["app"] = app_module


_bootstrap_local_app_package()

from app.schema.policy_chatbot_schema import PolicyChatbotAskRequest, UserProfileInput
from app.service.policy_chatbot.chat_app import (
    answer_policy_chatbot_query,
    build_retrieval_query,
    load_service_intro_document,
)


st.set_page_config(page_title="Policy Chatbot Test", layout="wide")
st.title("Policy Chatbot Streamlit Test")

st.caption("질문 + 사용자 정보를 넣고 policy chatbot 응답을 바로 테스트합니다.")

query = st.text_area(
    "질문",
    value="한식점 창업하려고 하는데 2%대 저금리 대출 있을까?",
    height=120,
)

gu_options = ["미입력", "금천구", "영등포구", "구로구", "관악구", "동작구", "서초구", "강남구", "마포구", "용산구", "성동구", "광진구", "중구", "종로구", "성북구", "강북구", "도봉구", "노원구", "은평구", "강서구", "양천구", "중랑구", "동대문구", "서대문구", "강동구", "송파구"]
col1, col2, col3, col4, col5 = st.columns(5)
industry = col1.text_input("업종", value="한식음식점")
age_input = col2.text_input("나이", value="35")
biz = col3.selectbox("사업자등록여부", options=["미입력", "있음", "없음"], index=1)
gu = col4.selectbox("지역", options=gu_options, index=1)
business_stage = col5.selectbox(
    "창업 상태",
    options=["미입력", "예비창업", "창업", "재창업"],
    index=2,
)

session_idx_input = st.text_input("session_idx (선택)", value="")

if st.button("테스트 실행", type="primary"):
    try:
        age = int(age_input) if age_input.strip() else None
    except ValueError:
        st.error("나이는 숫자로 입력해주세요.")
        st.stop()

    if biz == "있음":
        has_business_registration = True
    elif biz == "없음":
        has_business_registration = False
    else:
        has_business_registration = None

    user_profile = UserProfileInput(
        industry=industry or None,
        age=age,
        has_business_registration=has_business_registration,
        business_stage=business_stage if business_stage != "미입력" else None,
        startup_status=business_stage if business_stage != "미입력" else None,
        region=gu if gu != "미입력" else None,
    )

    profile_dict = user_profile.model_dump() if user_profile else None
    retrieval_query = build_retrieval_query(query.strip(), profile_dict)

    st.subheader("build_retrieval_query 결과")
    st.code(retrieval_query)


    request = PolicyChatbotAskRequest(
        user_query=query.strip(),
        user_profile=user_profile,
        session_idx=int(session_idx_input) if session_idx_input.strip().isdigit() else None,
    )

    with st.spinner("답변 생성 중..."):
        started_at = time.perf_counter()
        result = answer_policy_chatbot_query(request)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    st.subheader("답변")
    st.write(result.get("answer", ""))
    st.caption(f"응답 시간: {elapsed_ms} ms ({elapsed_ms / 1000:.2f}초)")

    st.subheader("메타")
    st.json(
        {
            "type": result.get("type"),
            "model": result.get("model"),
            "session_title_suggestion": result.get("session_title_suggestion"),
            "latency": result.get("latency"),
        }
    )

    st.subheader("검색 문서")
    st.json(result.get("retrieved_documents", []))

    if result.get("type") == "service_intro":
        original_intro_doc = load_service_intro_document("./app/service/policy_chatbot/original_data")
        if original_intro_doc:
            st.subheader("서비스 소개 원본")
            st.text_area(
                "잘리지 않은 원문",
                value=original_intro_doc.get("content", ""),
                height=600,
            )

    with st.expander("원본 전체 응답(JSON)"):
        st.json(result)
