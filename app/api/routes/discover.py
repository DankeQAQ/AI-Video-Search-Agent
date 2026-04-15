import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.schemas.interpreter import InterpretedIntent
from app.schemas.metadata import TmdbEnrichment
from app.schemas.search import SearchResponse
from app.services.metadata import fetch_tmdb_enrichment
from app.services.search_engine import search_watch_candidates

logger = logging.getLogger(__name__)

router = APIRouter()


class TmdbLookupRequest(BaseModel):
    title: str = Field(..., min_length=1)
    original_title: Optional[str] = None
    media_type: Optional[str] = None
    season: Optional[int] = Field(None, ge=1)


@router.post("/tmdb", response_model=TmdbEnrichment, summary="TMDB 元数据对齐")
async def lookup_tmdb(body: TmdbLookupRequest) -> TmdbEnrichment:
    try:
        return await fetch_tmdb_enrichment(
            body.title,
            original_title=body.original_title,
            media_type=body.media_type,
            season_number=body.season,
        )
    except RuntimeError as e:
        logger.warning("TMDB 路由返回 503: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("TMDB 路由未捕获异常: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e


class SearchRunRequest(BaseModel):
    intent: InterpretedIntent
    metadata: TmdbEnrichment
    overseas_mode: bool = Field(
        default=False,
        description="海外无忧模式：优化检索词并重排/标注结果（如澳洲等海外 IP）。",
    )


@router.post("/search", response_model=SearchResponse, summary="搜索观看候选链接")
async def run_search(body: SearchRunRequest) -> SearchResponse:
    try:
        return await search_watch_candidates(
            body.intent,
            body.metadata,
            overseas_mode=body.overseas_mode,
        )
    except RuntimeError as e:
        logger.warning("搜索路由返回 503: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("搜索路由未捕获异常: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e
