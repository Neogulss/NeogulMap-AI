import numpy as np
import pandas as pd
import xgboost as xgb
import logging

from sales_model_loader import sales_models
from schemas import SalesInput, SalesOutput
from exceptions import PredictionError

# 예외 처리
logger = logging.getLogger(__name__)

config = sales_models["config"]
cat_map = sales_models["category_mappings"]
kmeans = sales_models["kmeans"]
lgb_clf = sales_models["lgb_clf"]
lgb_reg = sales_models["lgb_reg"]
xgb_reg = sales_models["xgb_reg"]

# 소규모 점포 기준: 유사_업종_점포_수 < 4이면 모델 예측 신뢰도 낮음
# → 학습 시 소규모 점포는 폐업률 노이즈가 심해 제외했던 기준과 동일
SMALL_STORE_THRESHOLD = config["small_store_threshold"]

# 앙상블 가중치: 학습 시 최적 가중치 탐색 결과
# → LGB 소프트 0.25 + XGB 0.75 조합이 MdAPE 기준 최적
LGB_WEIGHT = config["best_w_soft"]       # 0.25
XGB_WEIGHT = 1 - config["best_w_soft"]  


# 전처리 함수
def _encode_input(data: SalesInput) -> pd.DataFrame:
    """
    SalesInput → 모델 입력용 DataFrame 변환

    - 행정동_코드, 서비스_업종_코드를 학습 때 사용한 인덱스로 변환
      (sales_category_mappings.json 기준)
    - config["features"] 순서대로 컬럼 정렬
      (모델은 학습 때와 동일한 피처 순서를 요구함)
    - 없는 값은 NaN → LightGBM/XGBoost가 NaN 그대로 처리
    """

    row = data.model_dump()

    # 카테고리 인코딩
    # 학습 때 {값: 인덱스} 형태로 변환했던 것과 동일하게 적용
    # 매핑에 없는 값이면 -1 (미등록 행정동/업종)
    row["행정동_코드"] = cat_map["행정동_코드"].get(row["행정동_코드"], -1)
    row["서비스_업종_코드"] = cat_map["서비스_업종_코드"].get(row["서비스_업종_코드"], -1)

    # 피처 순서 맞춰서 DataFrame 생성
    df = pd.DataFrame([row])[config["features"]]
    
    # categorical 타입 지정
    for col in config["cat_cols"]:
        df[col] = df[col].astype("category")
    
    return df


# 예측 메인 함수
def predict_sales(data: SalesInput) -> SalesOutput:
    """
    매출 예측 전체 파이프라인

    [흐름]
    1. 소규모 점포 예외처리
    2. 입력 전처리
    3. K-means로 구간 결정
    4. LGB 소프트 보팅 예측
    5. XGB 확정 구간 예측
    6. 앙상블 → expm1 역변환
    7. 하한/상한 벗어나면 LOW 반환
    8. 정상이면 HIGH 반환
    """
    try:
        # STEP 1. 소규모 점포 예외처리
        
        # 유사_업종_점포_수 < 4이면 통계적으로 불안정
        # → 모델 예측 없이 바로 LOW 반환
        if data.유사_업종_점포_수 < SMALL_STORE_THRESHOLD:
            return SalesOutput(
                pred_sales=0.0,
                segment=-1,
                confidence="LOW",
                message="점포 수가 적어 예측 신뢰도가 낮습니다."
            )

        # STEP 2. 입력 전처리
        X = _encode_input(data)

        # STEP 3. K-means 구간 결정
        
        # K-means는 학습 때 log 스케일 매출 기준으로 4구간으로 나눴음
        # → 예측 시에도 동일하게 log 매출 기준으로 구간 결정
        # cluster_mapping: K-means 원본 번호 → 우리 구간 번호 (저→고 순서)
        log_sales = X[["sales_lag1_log"]].fillna(0).values
        raw_cluster = int(kmeans.predict(log_sales)[0])
        cluster_mapping = {int(k): v for k, v in config["cluster_mapping"].items()}
        segment = cluster_mapping.get(raw_cluster, raw_cluster)

        # STEP 4. LightGBM 소프트 보팅 예측

        # lgb_clf: 4개 구간에 속할 확률 반환 (합계 = 1.0)
        # 소프트 보팅: 각 구간 예측값 × 해당 구간 확률 → 가중합
        # → 구간 경계 근처일 때 예측값이 급격히 튀지 않도록 완화
        lgb_proba = lgb_clf.predict(X)[0]  # shape: (4,)
        lgb_pred_log = sum(
            lgb_proba[seg] * lgb_reg[seg].predict(X)[0]
            for seg in range(4)
        )

        # STEP 5. XGBoost 확정 구간 예측
    
        # XGBoost는 K-means로 확정된 구간의 모델만 사용
        # (소프트 보팅 없이 단일 구간 예측)
        xgb_input = xgb.DMatrix(X, enable_categorical=True)
        xgb_pred_log = float(xgb_reg[segment].predict(xgb_input)[0])

        # STEP 6. 앙상블 + 역변환
        
        # log 스케일에서 가중 평균 후 expm1으로 원금액 복원
        # LGB 0.25 + XGB 0.75 (학습 시 최적 가중치)
        ensemble_log = LGB_WEIGHT * lgb_pred_log + XGB_WEIGHT * xgb_pred_log
        pred_sales = float(np.expm1(ensemble_log))
        
        # STEP 7. 하한/상한 예외처리
        
        # 학습 범위 밖 → 모델이 본 적 없는 데이터 → LOW
        # lower_bound: 100만원 (학습 시 하한 필터 기준)
        # upper_bound: 374,179,397원 (학습 시 p99.9 상한 필터 기준)
        if pred_sales < config["lower_bound"]:
            return SalesOutput(
                pred_sales=pred_sales,
                segment=segment,
                confidence="LOW",
                message="예측 매출이 너무 낮아(100만원 미만) 신뢰도가 낮습니다."
            )

        if pred_sales > config["upper_bound"]:
            return SalesOutput(
                pred_sales=pred_sales,
                segment=segment,
                confidence="LOW",
                message="고매출 특수상권(4억 이상)으로 예측 신뢰도가 낮습니다."
            )

        # STEP 8. 정상 반환
        return SalesOutput(
            pred_sales=pred_sales,
            segment=segment,
            confidence="HIGH",
            message=None
        )
    except PredictionError:
        raise
    except KeyError as e:
        logger.exception("Config or mapping key missing: %s", e)
        raise PredictionError("모델 설정이 올바르지 않습니다.")
    except FileNotFoundError as e:
        logger.exception("Model file missing: %s", e)
        raise PredictionError("예측 모델 파일을 찾을 수 없습니다.")
    except Exception:
        logger.exception("Unexpected error during prediction. input=%s", data.model_dump())
        raise PredictionError("매출 예측 처리 중 오류가 발생했습니다.")