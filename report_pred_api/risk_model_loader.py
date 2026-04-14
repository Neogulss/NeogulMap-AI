import os
import json
import lightgbm as lgb
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

# 모델 파일 경로
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_risk_models")


def load_risk_models():
    """
    폐업 리스크 예측 모델 전체 로드
    반환: {
        "config": dict,
        "bayesian_encoding": dict,
        "lgb_clf": lgb.Booster,
        "xgb_reg": xgb.Booster,
    }
    """
    try:
        models = {}

        # config
        with open(os.path.join(MODEL_DIR, "risk_model_config.json"), "r", encoding="utf-8") as f:
            models["config"] = json.load(f)

        # 베이지안 인코딩 매핑
        with open(os.path.join(MODEL_DIR, "risk_bayesian_encoding.json"), "r", encoding="utf-8") as f:
            models["bayesian_encoding"] = json.load(f)

        # LightGBM 분류 (1단계)
        models["lgb_clf"] = lgb.Booster(
            model_file=os.path.join(MODEL_DIR, "risk_lgb_clf.txt")
        )

        # XGBoost 회귀 (2단계)
        booster = xgb.Booster()
        booster.load_model(os.path.join(MODEL_DIR, "risk_xgb_reg.json"))
        models["xgb_reg"] = booster

        logger.info("폐업 리스크 예측 모델 로드 완료")
        return models
    except Exception:
        logger.exception("폐업 리스크 예측 모델 로드 실패")
        raise


# 앱 시작 시 한 번만 로드
risk_models = load_risk_models()
