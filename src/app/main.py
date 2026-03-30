from fastapi import FastAPI
#from src.app.api.v1 import api

app = FastAPI()

# 라우터 등록
#app.include_router(api.api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "AI Server Running"}