import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# ── 모델 로드 ──────────────────────────────────────────────
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
    estimatedCost:           Optional[int]  # 권리금 + 보증금 합산 (만원)


class RecommendResponse(BaseModel):
    results: List[DongResult]


# ── 유틸 함수 ─────────────────────────────────────────────
def get_area_bin(면적):
    if 면적 <= 30:    return '~30㎡'
    elif 면적 <= 50:  return '31~50㎡'
    elif 면적 <= 80:  return '51~80㎡'
    elif 면적 <= 120: return '81~120㎡'
    elif 면적 <= 200: return '121~200㎡'
    else:             return '200㎡~'


def get_floor_bin(층수):
    if 층수 < 0:    return '지하'
    elif 층수 == 1: return '1층'
    else:           return '2층이상'


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

    if req.service_type not in enc_boseo_100:
        return RecommendResponse(results=[])

    budget      = req.budget
    budget_low  = budget * 0.7
    budget_high = budget * 1.0
    면적구간     = get_area_bin(req.area)
    층수구간     = get_floor_bin(req.floor)
    enc_b       = enc_boseo_100.get(req.service_type, mean_boseo)
    enc_k       = enc_kwon_100.get(req.service_type, mean_kwon)
    업종코드     = upjong_code_dict.get(req.service_type)

    candidates = []

    for hdong in xval['hdong'].dropna().unique():
        if hdong not in xd.index:
            continue

        vb = build_vec(hdong, feats_boseo, enc_b,
                       '업종_enc_보증금', req.floor, req.area)
        vk = build_vec(hdong, feats_kwon,  enc_k,
                       '업종_enc_권리금', req.floor, req.area)
        if vb is None or vk is None:
            continue

        pred_b     = max(0,   round(model_boseo.predict(vb)[0] / 100) * 100)
        pred_k     = max(500, round(model_kwon.predict(vk)[0]  / 100) * 100)
        pred_total = pred_b + pred_k

        # 실거래 통계 보정
        실거래 = stats_detail[
            (stats_detail['hdong']     == hdong) &
            (stats_detail['서울시_업종'] == req.service_type) &
            (stats_detail['층수_구간']  == 층수구간) &
            (stats_detail['면적_구간']  == 면적구간)
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
            adminDongName           = c['hdong'],
            districtCode            = district_code,
            districtName            = district_name,
            longitude               = info.get('longitude'),
            latitude                = info.get('latitude'),
            serviceIndustryCode     = 업종코드,
            serviceIndustryCodeName = req.service_type,
            estimatedCost           = c['display'],
        ))

    return RecommendResponse(results=results)