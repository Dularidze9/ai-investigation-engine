import logging
import pathlib

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api import documents, health, query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC = pathlib.Path(__file__).parent / "static"

app = FastAPI(title="AI Investigation Engine")
app.include_router(health.router)
app.include_router(query.router)
app.include_router(documents.router)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
