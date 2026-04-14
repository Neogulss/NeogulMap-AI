import os
import json
import pickle
import lightgbm as lgb
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

# 모델 파일 경로
# MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "saved_models")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_sales_models")

def load_sales_models():
    """
    매출 예측 모델 전체 로드
    반환: {
        "config": dict,
        "category_mappings": dict,
        "kmeans": KMeans,
        "lgb_clf": lgb.Booster,
        "lgb_reg": {0: lgb.Booster, 1: ..., 2: ..., 3: ...},
        "xgb_reg": {0: xgb.Booster, 1: ..., 2: ..., 3: ...},
    }
    """
    try: 
        # 데이터 담아갈 모델 선언
        models = {}

        # config
        with open(os.path.join(MODEL_DIR, "sales_model_config.json"), "r", encoding="utf-8") as f:
            models["config"] = json.load(f)

        # 카테고리 매핑
        with open(os.path.join(MODEL_DIR, "sales_category_mappings.json"), "r", encoding="utf-8") as f:
            raw = json.load(f)
            # 행정동_코드: 값 → 인덱스 딕셔너리로 변환
            models["category_mappings"] = {
                "행정동_코드": {v: i for i, v in enumerate(raw["행정동_코드"])},
                "서비스_업종_코드": {v: i for i, v in enumerate(raw["서비스_업종_코드"])},
            }

        # K-Means
        with open(os.path.join(MODEL_DIR, "kmeans.pkl"), "rb") as f:
            models["kmeans"] = pickle.load(f)

        # LightGBM 분류 (1단계)
        models["lgb_clf"] = lgb.Booster(
            model_file=os.path.join(MODEL_DIR, "lgb_clf.txt")
        )

        # LightGBM 회귀 (2단계, 구간별)
        models["lgb_reg"] = {}
        for seg in range(4):
            models["lgb_reg"][seg] = lgb.Booster(
                model_file=os.path.join(MODEL_DIR, f"lgb_reg_{seg}.txt")
            )

        # XGBoost 회귀 (2단계, 구간별)
        models["xgb_reg"] = {}
        for seg in range(4):
            booster = xgb.Booster()
            booster.load_model(os.path.join(MODEL_DIR, f"xgb_reg_{seg}.json"))
            models["xgb_reg"][seg] = booster

        logger.info("매출 예측 모델 로드 완료")
        return models
    except Exception:
        logger.exception("매출 예측 모델 로드 실패")
        raise


# 앱 시작 시 한 번만 로드
sales_models = load_sales_models()