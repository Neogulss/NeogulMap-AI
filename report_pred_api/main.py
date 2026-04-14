import logging

from fastapi import FastAPI, HTTPException
from .schemas import SalesInput, SalesOutput
from .sales_predictor import predict_sales
from .exceptions import PredictionError
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s %(levelname)s %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="REPORT PRED API",
    description="sales_per_month pred model FAST_API",
    version="1.0.0" # 변경 시 docs를 위해 version 변경
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