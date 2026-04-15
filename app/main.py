from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.router import api_router

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="视频聚合搜索AI Agent",
    description="基于 PRD 的 FastAPI 后端：意图解析与后续检索能力。",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health", summary="根路径存活探针（负载均衡常用）")
async def root_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", summary="控制台页面", include_in_schema=False)
async def index_page():
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"service": "video-search-agent", "docs": "/docs", "api": "/api/v1", "hint": "缺少 static/index.html"}
