import logging
import shap

import numpy as np
import pandas as pd
import xgboost as xgb

from exceptions import PredictionError
from risk_model_loader import risk_models
from schemas import RiskInput, RiskOutput


logger = logging.getLogger(__name__)

config = risk_models["config"]
bayesian_encoding = risk_models["bayesian_encoding"]
lgb_clf = risk_models["lgb_clf"]
xgb_reg = risk_models["xgb_reg"]

# SHAP
risk_explainer = shap.TreeExplainer(lgb_clf)

SMALL_STORE_THRESHOLD = config["small_store_filter"]
CLASSIFICATION_THRESHOLD = config["threshold"]
FP_PROB_THRESHOLD = config["fp_prob_threshold"]
PRED_VAL_THRESHOLD = config["pred_val_threshold"]
GLOBAL_MEAN = bayesian_encoding.get(
    "global_mean",
    config.get("bayesian_global_mean", 0.0),
)


def _encode_input(data: RiskInput) -> pd.DataFrame:
    """
    RiskInput -> 모델 입력용 DataFrame 변환

    - 행정동/업종 원본 키로 베이지안 인코딩 파생 피처 생성
    - config["features"] 순서에 맞춰 컬럼 정렬
    - 학습 산출물에 결측 대체값이 저장돼 있지 않아 NaN은 그대로 유지
    """
    row = data.model_dump()
    
    change_map = {
        "HH": 0,
        "HL": 1,
        "LH": 2,
        "LL": 3,
    }

    row["상권_변화_지표_encoded"] = change_map.get(row["상권_변화_지표"], -1)
    row["above_survival_avg"] = 1 if row["상권_변화_지표"][0] == "H" else 0
    row["above_closure_avg"] = 1 if row["상권_변화_지표"][1] == "H" else 0


    dong_code = str(row["행정동_코드"])
    biz_code = str(row["서비스_업종_코드"])
    dong_biz_key = f"{dong_code}_{biz_code}"

    row["dong_smoothed"] = bayesian_encoding["dong_map"].get(dong_code, GLOBAL_MEAN)
    row["biz_smoothed"] = bayesian_encoding["biz_map"].get(biz_code, GLOBAL_MEAN)
    row["dong_biz_smoothed"] = bayesian_encoding["dong_biz_map"].get(
        dong_biz_key,
        GLOBAL_MEAN,
    )

    df = pd.DataFrame([row])[config["features"]]
    return df


def _get_top_risk_factors(X: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    LightGBM 분류모델 기준 SHAP 상위 기여 피처 반환

    반환 예시:
    [
        {
            "feature": "전분기_폐업_률",
            "feature_value": 4.2,
            "impact": 0.31,
            "direction": "up",
        }
    ]
    """
    shap_values = risk_explainer.shap_values(X)

    # shap 버전/모델에 따라 반환 형식이 달라질 수 있어 단일 행 기준으로 정규화
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[-1])[0]
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[0, :, -1]
        elif values.ndim == 2:
            values = values[0]
        else:
            values = np.ravel(values)

    factors = []
    for feature, impact in zip(config["features"], values):
        feature_value = X.iloc[0][feature]
        factors.append(
            {
                "feature": feature,
                "feature_value": None if pd.isna(feature_value) else float(feature_value) if isinstance(feature_value, (np.floating, float, int, np.integer)) else feature_value,
                "impact": float(impact),
                "direction": "up" if impact > 0 else "down",
            }
        )

    factors.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return factors[:top_n]


def predict_risk(data: RiskInput) -> RiskOutput:
    """
    폐업 리스크 예측 전체 파이프라인

    [흐름]
    1. 소규모 점포 예외처리
    2. 베이지안 인코딩 포함 입력 전처리
    3. LightGBM 분류 확률 계산
    4. threshold 이상일 때만 XGBoost 회귀 수행
    5. 확률/예측값 후처리
    6. 최종 리스크 레벨과 메시지 반환
    """
    try:
        if data.유사_업종_점포_수 < SMALL_STORE_THRESHOLD:
            return RiskOutput(
                risk_prob=0.0,
                risk_closure_rate=0.0,
                risk_level="LOW",
                # SHAP 결과
                top_risk_factors=None,
                message="점포 수가 적어 폐업 위험 예측 신뢰도가 낮습니다.",
            )

        X = _encode_input(data)

        risk_prob = float(lgb_clf.predict(X)[0])
        pred_cls = int(risk_prob >= CLASSIFICATION_THRESHOLD)
        
        # SHAP 결과 top N 정리
        top_risk_factors = _get_top_risk_factors(X)

        risk_closure_rate = 0.0
        if pred_cls == 1:
            xgb_input = xgb.DMatrix(X)
            pred_log = float(xgb_reg.predict(xgb_input)[0])
            risk_closure_rate = float(np.expm1(pred_log))

        # 학습 시 저장한 후처리 규칙을 그대로 적용
        if risk_prob < FP_PROB_THRESHOLD:
            risk_closure_rate = 0.0

        if risk_closure_rate < PRED_VAL_THRESHOLD:
            risk_closure_rate = 0.0

        risk_level = "HIGH" if risk_closure_rate > 0 else "LOW"
        message = None

        if pred_cls == 1 and risk_closure_rate == 0.0:
            message = "폐업 위험 신호는 있으나 최종 후처리 기준 미만으로 판단되었습니다."

        return RiskOutput(
            risk_prob=risk_prob,
            risk_closure_rate=risk_closure_rate,
            risk_level=risk_level,
            # SHAP 결과
            top_risk_factors=top_risk_factors,
            message=message,
        )
    except PredictionError:
        raise
    except KeyError as e:
        logger.exception("Config or encoding key missing: %s", e)
        raise PredictionError("폐업 예측 모델 설정이 올바르지 않습니다.")
    except FileNotFoundError as e:
        logger.exception("Model file missing: %s", e)
        raise PredictionError("폐업 예측 모델 파일을 찾을 수 없습니다.")
    except Exception:
        logger.exception("Unexpected error during risk prediction. input=%s", data.model_dump())
        raise PredictionError("폐업 위험 예측 처리 중 오류가 발생했습니다.")
