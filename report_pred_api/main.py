import logging

from fastapi import FastAPI, HTTPException
from schemas import SalesInput, SalesOutput, RiskInput, RiskOutput
from sales_predictor import predict_sales
from risk_predictor import predict_risk
from exceptions import PredictionError
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s %(levelname)s %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="REPORT PRED API",
    description="sales and risk prediction model FAST_API",
    version="2.0.0" # 변경 시 docs를 위해 version 변경
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/report_pred/sales/health")
def health_check():
    return {"status": "sales_pred_api ok"}


@app.post("/report_pred/sales", response_model=SalesOutput)
def predict_sales_endpoint(data: SalesInput):
    try:
        return predict_sales(data)
    except PredictionError as e:
        logger.warning("Prediction failed: %s", e.public_message)
        raise HTTPException(status_code=503, detail=e.public_message)
    except Exception:
        logger.exception("Unhandled error in /report_pred/sales")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")
    

@app.get("/report_pred/risk/health")
def risk_health_check():
    return {"status" : "risk_pred_api ok"}


@app.post("/report_pred/risk", response_model=RiskOutput)
def predict_risk_endpoint(data: RiskInput):
    try:
        return predict_risk(data)
    except PredictionError as e:
        logger.warning("Prediction failed: %s", e.public_message)
        raise HTTPException(status_code=503, detail=e.public_message)
    except Exception:
        logger.exception("Unhandled error in /report_pred/sales")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")