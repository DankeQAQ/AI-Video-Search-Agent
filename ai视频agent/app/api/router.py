from fastapi import APIRouter

from app.api.routes import discover, health, intent

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(intent.router)
api_router.include_router(discover.router)
