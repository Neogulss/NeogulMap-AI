from pydantic import BaseModel
from typing import Optional

# 폐업 예측 입력
class RiskInput(BaseModel):
    기준_년분기_코드: int
    기준_분기_코드: int
    행정동_코드: int
    서비스_업종_코드: str
    상권_변화_지표: str   # HH / HL / LH / LL
    점포_수: int
    유사_업종_점포_수: int
    개업_율: Optional[float] = None
    franchise_ratio: Optional[float] = None
    경쟁_밀도: Optional[float] = None
    영역_면적: Optional[float] = None
    월_평균_소득_금액: Optional[float] = None
    음식_지출_비율: Optional[float] = None
    유흥_지출_비율: Optional[float] = None
    교육_지출_비율: Optional[float] = None
    여가문화_지출_비율: Optional[float] = None
    log_총_유동인구_수: Optional[float] = None
    floating_population_per_store: Optional[float] = None
    young_pop_ratio: Optional[float] = None
    weekend_pop_ratio: Optional[float] = None
    운영_영업_개월_평균: Optional[float] = None
    폐업_영업_개월_평균: Optional[float] = None
    op_months_vs_closure_avg: Optional[float] = None
    operation_months_vs_seoul_avg: Optional[float] = None
    avg_sales_per_transaction: Optional[float] = None
    weekend_sales_ratio: Optional[float] = None
    sales_missing_flag: Optional[int] = None
    sales_missing_type_code: Optional[int] = None
    전분기_폐업_률: Optional[float] = None
    폐업_률_변화_량: Optional[float] = None
    유사_업종_점포_수_변화_량: Optional[float] = None
    유동인구_변화_량: Optional[float] = None
    개업_율_lag1: Optional[float] = None
    avg_sales_per_transaction_lag1: Optional[float] = None
    # above_survival_avg: Optional[int] = None
    # above_closure_avg: Optional[int] = None
    

# 폐업 예측 출력
class RiskOutput(BaseModel):
    risk_prob: float                  # 폐업 위험 확률 (0~1)
    risk_closure_rate: float                 # 회귀 기반 위험 점수
    risk_level: str                   # EX) LOW / MEDIUM / HIGH
    # confidence: str                   # 예측 신뢰도 : HIGH / LOW
    top_risk_factors: Optional[list[dict]] = None
    message: Optional[str] = None     # 예외/주의 안내 문구
    

# 매출 예측 입력
class SalesInput(BaseModel):
    # 모델 feature(.csv 기반)
    기준_분기_코드: int
    행정동_코드: int
    서비스_업종_코드: str
    sales_lag1_log: Optional[float] = None
    sales_lag2_log: Optional[float] = None
    sales_lag3_log: Optional[float] = None
    sales_lag4_log: Optional[float] = None
    sales_ma2: Optional[float] = None
    sales_ma3: Optional[float] = None
    sales_std2: Optional[float] = None
    sales_std3: Optional[float] = None
    sales_growth_1q: Optional[float] = None
    sales_vs_ma3: Optional[float] = None
    매출_증감률: Optional[float] = None
    매출_대비_업종평균_비율: Optional[float] = None
    점포_수: int
    유사_업종_점포_수: int
    franchise_ratio: Optional[float] = None
    경쟁_밀도: Optional[float] = None
    competition_ratio: Optional[float] = None
    영역_면적: Optional[float] = None
    floating_population_per_store: Optional[float] = None
    유동인구_밀도: Optional[float] = None
    young_pop_ratio: Optional[float] = None
    weekend_pop_ratio: Optional[float] = None
    log_총_유동인구_수: Optional[float] = None
    월_평균_소득_금액: Optional[float] = None
    음식_지출_비율: Optional[float] = None
    유흥_지출_비율: Optional[float] = None
    교육_지출_비율: Optional[float] = None
    여가문화_지출_비율: Optional[float] = None
    유사_업종_점포_수_lag1: Optional[float] = None
    유사_업종_점포_수_변화_량: Optional[float] = None
    폐업_률_변화_량: Optional[float] = None
    총_유동인구_수_lag1: Optional[float] = None
    유동인구_변화_량: Optional[float] = None


# 매출 예측 출력
class SalesOutput(BaseModel):
    pred_sales: float               # 예측 월 매출 (원)
    segment: int                    # K-means 구간 4개
    confidence: str                 # 예측 신뢰도 : HIGH / LOW(구간 미만, 구간 초과)
    top_sales_factors: Optional[list[dict]] = None  # feature importance TOP N 출력
    message: Optional[str] = None   # 소규모/대규모 예외 시 안내 메시지