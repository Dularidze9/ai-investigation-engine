from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.db import check_db

router = APIRouter()


@router.get("/health")
def health():
    try:
        return check_db()
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"db": "down", "detail": str(e)})
