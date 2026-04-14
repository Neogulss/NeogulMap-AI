from pydantic import BaseModel
from typing import Optional

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
    message: Optional[str] = None   # 소규모/대규모 예외 시 안내 메시지