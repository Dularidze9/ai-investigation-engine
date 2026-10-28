import logging

from fastapi import FastAPI

from app.api import health, query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="AI Investigation Engine")
app.include_router(health.router)
app.include_router(query.router)
