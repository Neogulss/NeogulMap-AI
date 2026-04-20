import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# ── CORS ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",       # Vite 개발 서버
        "http://13.209.5.156",         # EC2 운영
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── 모델 및 데이터 로드 ────────────────────────────────────
with open('행정동_추천_모델.pkl', 'rb') as f:
    payload = pickle.load(f)

model_boseo      = payload['model_boseo']
model_kwon       = payload['model_kwon']
feats_boseo      = payload['feats_boseo']
feats_kwon       = payload['feats_kwon']
enc_boseo_100    = payload['enc_boseo_100']
enc_kwon_100     = payload['enc_kwon_100']
mean_boseo       = payload['mean_boseo']
mean_kwon        = payload['mean_kwon']
xd               = payload['xd']
xval             = payload['xval']
score_dict       = payload['score_dict']
stats_detail     = payload['stats_detail']
stats_upjong     = payload['stats_upjong']
upjong_code_dict = payload['upjong_code_dict']
SEOUL_TO_JOMPO   = payload['SEOUL_TO_JOMPO']

# 깨진 행정동명 복원 딕셔너리
DONG_FIX = {
    '금호2?3가동':      '금호2·3가동',
    '면목3?8동':        '면목3·8동',
    '상계3?4동':        '상계3·4동',
    '상계6?7동':        '상계6·7동',
    '종로1?2?3?4가동':  '종로1·2·3·4가동',
    '종로5?6가동':      '종로5·6가동',
    '중계2?3동':        '중계2·3동',
}

def fix_dong_name(name):
    return DONG_FIX.get(name, name)

# 행정동 좌표 CSV
coord_df   = pd.read_csv('행정동_좌표.csv', encoding='utf-8-sig')
coord_dict = coord_df.set_index('hdong').to_dict('index')

print(f"모델 로드 완료 | 행정동 {len(coord_dict)}개")


# ── 스키마 ─────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    service_type: str           # 서울시 업종명 (예: 커피-음료)
    floor:        int           # 층수
    area:         float         # 면적 (㎡)
    budget:       int           # 초기자본금 (만원)
    top_n:        Optional[int] = 10


class DongResult(BaseModel):
    adminDongCode:           Optional[str]
    adminDongName:           str
    districtCode:            Optional[str]
    districtName:            Optional[str]
    longitude:               Optional[float]
    latitude:                Optional[float]
    serviceIndustryCode:     Optional[str]
    serviceIndustryCodeName: Optional[str]
    estimatedCost:           Optional[int]


class RecommendResponse(BaseModel):
    results: List[DongResult]


# ── 유틸 함수 ─────────────────────────────────────────────
def get_area_bin(area):
    if area <= 30:    return '~30㎡'
    elif area <= 50:  return '31~50㎡'
    elif area <= 80:  return '51~80㎡'
    elif area <= 120: return '81~120㎡'
    elif area <= 200: return '121~200㎡'
    else:             return '200㎡~'


def get_floor_bin(floor):
    if floor < 0:    return '지하'
    elif floor == 1: return '1층'
    else:            return '2층이상'


def build_vec(hdong, feats, enc_val, enc_col, floor, area):
    if hdong not in xd.index:
        return None
    xrow = xd.loc[hdong]
    row  = {
        enc_col:        enc_val,
        '층수_숫자':    floor,
        '1층여부':     1 if floor == 1 else 0,
        '면적_숫자_L':  np.log1p(area),
    }
    for f in feats:
        if f in row:
            continue
        raw = f.replace('_L', '')
        if f.endswith('_L') and raw in xrow.index:
            row[f] = np.log1p(max(0, xrow[raw]))
        elif f in xrow.index:
            row[f] = xrow[f]
        else:
            row[f] = 0
    return pd.DataFrame([row])[feats].fillna(0)


# ── 엔드포인트 ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):

    # 업종 유효성 확인
    if req.service_type not in enc_boseo_100:
        return RecommendResponse(results=[])

    budget      = req.budget
    budget_low  = budget * 0.6
    budget_high = budget * 1.4
    area_bin    = get_area_bin(req.area)
    floor_bin   = get_floor_bin(req.floor)
    enc_b       = enc_boseo_100.get(req.service_type, mean_boseo)
    enc_k       = enc_kwon_100.get(req.service_type, mean_kwon)
    industry_code = upjong_code_dict.get(req.service_type)

    candidates = []

    for hdong in xval['hdong'].dropna().unique():
        if hdong not in xd.index:
            continue

        # 모델 예측
        vb = build_vec(hdong, feats_boseo, enc_b,
                       '업종_enc_보증금', req.floor, req.area)
        vk = build_vec(hdong, feats_kwon,  enc_k,
                       '업종_enc_권리금', req.floor, req.area)
        if vb is None or vk is None:
            continue

        pred_b     = max(0,   round(model_boseo.predict(vb)[0] / 100) * 100)
        pred_k     = max(500, round(model_kwon.predict(vk)[0]  / 100) * 100)
        pred_total = pred_b + pred_k

        # 실거래 통계 보정 (상세 → 기본 fallback)
        실거래 = stats_detail[
            (stats_detail['hdong']     == hdong) &
            (stats_detail['서울시_업종'] == req.service_type) &
            (stats_detail['층수_구간']  == floor_bin) &
            (stats_detail['면적_구간']  == area_bin)
        ]
        if 실거래.empty:
            실거래 = stats_upjong[
                (stats_upjong['hdong']     == hdong) &
                (stats_upjong['서울시_업종'] == req.service_type)
            ]

        if not 실거래.empty and 실거래.iloc[0]['거래건수'] >= 3:
            r       = 실거래.iloc[0]
            display = int((pred_total + r['중앙값']) / 2 / 100) * 100
        else:
            display = pred_total

        # 예산 범위 안 우선, 밖이면 차이 작은 순
        if budget_low <= display <= budget_high:
            priority = 0
            diff     = budget - display
        else:
            priority = 1
            diff     = abs(display - budget)

        candidates.append({
            'hdong':    hdong,
            'display':  display,
            'score':    score_dict.get(hdong, 0),
            'priority': priority,
            'diff':     diff,
        })

    if not candidates:
        return RecommendResponse(results=[])

    # 예산 범위 안 → 상권점수 높은 순
    # 예산 범위 밖 → 차이 작은 순
    candidates.sort(key=lambda x: (
        x['priority'],
        -x['score'] if x['priority'] == 0 else x['diff']
    ))

    results = []
    for c in candidates[:req.top_n]:
        info          = coord_dict.get(c['hdong'], {})
        district_code = str(info.get('자치구_코드', ''))
        district_name = str(info.get('자치구명', ''))

        results.append(DongResult(
    adminDongCode           = str(info.get('행정동_코드', '')),
    adminDongName           = fix_dong_name(c['hdong']),  # ← 수정
    districtCode            = district_code,
    districtName            = district_name,
    longitude               = info.get('longitude'),
    latitude                = info.get('latitude'),
    serviceIndustryCode     = industry_code,
    serviceIndustryCodeName = req.service_type,
    estimatedCost           = c['display'],
))

    return RecommendResponse(results=results)