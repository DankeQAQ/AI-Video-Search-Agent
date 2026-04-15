"""
TMDB 元数据对齐：异步拉取海报、年份、多语言别名（含指定季的译名）。
文档：https://developer.themoviedb.org/docs

鉴权：支持 v3 短 API Key（query ?api_key=）与「API Read Access Token」长 JWT（Authorization: Bearer）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx

from app.core.config import get_settings
from app.schemas.metadata import TitleAlias, TmdbEnrichment

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(30.0)


def _year_from_date(date_str: Optional[str]) -> Optional[int]:
    if not date_str or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except ValueError:
        return None


def _poster_url(path: Optional[str], image_base: str) -> Optional[str]:
    if not path:
        return None
    return f"{image_base.rstrip('/')}/{path.lstrip('/')}"


def _parse_translation_block(
    block: Any,
    scope: Literal["series", "season"],
) -> List[TitleAlias]:
    """解析 TMDB translations 嵌套结构（append_to_response=translations）。"""
    out: List[TitleAlias] = []
    if block is None:
        return out
    if isinstance(block, dict) and "translations" in block:
        items = block.get("translations") or []
    elif isinstance(block, list):
        items = block
    else:
        return out
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        lang = (item.get("iso_639_1") or item.get("iso_639") or "und") or "und"
        data = item.get("data") or {}
        name = data.get("name") or data.get("title")
        if isinstance(name, str) and name.strip():
            out.append(TitleAlias(language=str(lang), title=name.strip(), scope=scope))
    return out


def _merge_unique_aliases(rows: List[TitleAlias]) -> List[TitleAlias]:
    seen = set()
    merged: List[TitleAlias] = []
    for r in rows:
        key = (r.language, r.title, r.scope)
        if key in seen:
            continue
        seen.add(key)
        merged.append(r)
    return merged


def _tmdb_auth_from_settings() -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    TMDB v3 接口两种凭证：
    - 短 v3 API Key：使用 query `api_key=...`
    - 长「API Read Access Token」（JWT，通常以 eyJ 开头）：必须使用 Header `Authorization: Bearer <token>`
    """
    settings = get_settings()
    key = (settings.tmdb_api_key or "").strip()
    if not key:
        raise RuntimeError("未设置 TMDB_API_KEY，无法请求 TMDB。")

    # JWT Read Access Token 或明显长 token → Bearer；否则按 v3 api_key 处理
    if key.startswith("eyJ") or len(key) > 40:
        logger.debug("TMDB 使用 Bearer Token 鉴权（key 长度=%s）", len(key))
        return {"Authorization": f"Bearer {key}"}, {}

    logger.debug("TMDB 使用 api_key 查询参数鉴权")
    return {}, {"api_key": key}


async def _tmdb_get(
    client: httpx.AsyncClient,
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    auth_headers, auth_params = _tmdb_auth_from_settings()
    q: Dict[str, Any] = {**auth_params, **(params or {})}
    try:
        r = await client.get(path, params=q, headers=auth_headers)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        snippet = (e.response.text or "")[:1200]
        logger.exception(
            "TMDB HTTP 错误 status=%s path=%s body[:800]=%r",
            e.response.status_code,
            path,
            snippet[:800],
        )
        raise RuntimeError(
            f"TMDB 返回 HTTP {e.response.status_code}（{path}）。"
            f"若使用长 Token，请确认 TMDB 账户里复制的是 API Read Access Token；"
            f"短 Key 请使用 v3 API Key。响应片段: {snippet[:280]}"
        ) from e
    except httpx.RequestError as e:
        logger.exception("TMDB 网络请求失败 path=%s: %s", path, e)
        raise RuntimeError(f"TMDB 网络错误（{path}）: {e}") from e

    try:
        data = r.json()
    except ValueError as e:
        logger.exception("TMDB 响应非 JSON path=%s text=%r", path, (r.text or "")[:500])
        raise RuntimeError(f"TMDB 返回非 JSON（{path}）") from e

    if not isinstance(data, dict):
        return {}
    return data


async def _search_best_tv(
    client: httpx.AsyncClient,
    query: str,
) -> Optional[Dict[str, Any]]:
    if not query.strip():
        return None
    data = await _tmdb_get(client, "/search/tv", {"query": query.strip()})
    results = data.get("results") or []
    if not results:
        return None
    return results[0]


async def _search_best_movie(
    client: httpx.AsyncClient,
    query: str,
) -> Optional[Dict[str, Any]]:
    if not query.strip():
        return None
    data = await _tmdb_get(client, "/search/movie", {"query": query.strip()})
    results = data.get("results") or []
    if not results:
        return None
    return results[0]


async def _top_cast_names(
    client: httpx.AsyncClient,
    tmdb_media: Literal["tv", "movie"],
    tmdb_id: int,
    *,
    limit: int = 3,
) -> List[str]:
    """取 credits.cast 中前几位演员姓名，供搜索词消歧。"""
    try:
        cr = await _tmdb_get(client, f"/{tmdb_media}/{tmdb_id}/credits")
    except RuntimeError as e:
        logger.warning("TMDB credits 跳过 id=%s: %s", tmdb_id, e)
        return []
    cast = cr.get("cast") or []
    names: List[str] = []
    if not isinstance(cast, list):
        return names
    for row in cast:
        if not isinstance(row, dict):
            continue
        n = row.get("name")
        if isinstance(n, str) and n.strip():
            t = n.strip()
            if t not in names:
                names.append(t)
        if len(names) >= limit:
            break
    return names


async def _pick_tmdb_entry(
    client: httpx.AsyncClient,
    title: str,
    original_title: Optional[str],
    media_type: Optional[str],
) -> Tuple[Optional[str], Optional[int]]:
    """
    返回 (tmdb_media, tmdb_id)。
    media_type: anime/tv -> 优先 TV；movie -> 优先电影。
    """
    mt = (media_type or "").lower()
    queries = [title.strip()]
    if original_title and original_title.strip() not in queries:
        queries.append(original_title.strip())

    async def try_tv() -> Optional[int]:
        for q in queries:
            hit = await _search_best_tv(client, q)
            if hit and hit.get("id") is not None:
                return int(hit["id"])
        return None

    async def try_movie() -> Optional[int]:
        for q in queries:
            hit = await _search_best_movie(client, q)
            if hit and hit.get("id") is not None:
                return int(hit["id"])
        return None

    if mt == "movie":
        mid = await try_movie()
        if mid is not None:
            return "movie", mid
        tid = await try_tv()
        if tid is not None:
            return "tv", tid
        return None, None

    tid = await try_tv()
    if tid is not None:
        return "tv", tid
    mid = await try_movie()
    if mid is not None:
        return "movie", mid
    return None, None


async def fetch_tmdb_enrichment(
    title: str,
    *,
    original_title: Optional[str] = None,
    media_type: Optional[str] = None,
    season_number: Optional[int] = None,
) -> TmdbEnrichment:
    """
    根据解析出的标题在 TMDB 中查找条目，返回海报 URL、年份、整剧与该季的多语言别名。
    """
    settings = get_settings()
    base = settings.tmdb_base_url.rstrip("/")
    image_base = settings.tmdb_image_base_url.rstrip("/")

    try:
        async with httpx.AsyncClient(base_url=base, timeout=DEFAULT_TIMEOUT) as client:
            tmdb_media, tmdb_id = await _pick_tmdb_entry(
                client, title, original_title, media_type
            )
            if tmdb_id is None or tmdb_media is None:
                return TmdbEnrichment(found=False)

            aliases: List[TitleAlias] = []

            if tmdb_media == "tv":
                tv = await _tmdb_get(
                    client,
                    f"/tv/{tmdb_id}",
                    {"append_to_response": "translations"},
                )
                primary = tv.get("name") or tv.get("original_name")
                poster_path = tv.get("poster_path")
                year = _year_from_date(tv.get("first_air_date"))

                trans_block = tv.get("translations")
                aliases.extend(_parse_translation_block(trans_block, "series"))

                alt = await _tmdb_get(client, f"/tv/{tmdb_id}/alternative_titles")
                for row in alt.get("results") or []:
                    t = row.get("title")
                    region = row.get("iso_3166_1") or "XX"
                    if isinstance(t, str) and t.strip():
                        aliases.append(
                            TitleAlias(
                                language=f"alt:{region}",
                                title=t.strip(),
                                scope="alternative",
                            )
                        )

                season_air_date: Optional[str] = None
                season_poster_path: Optional[str] = None
                if season_number is not None and season_number >= 1:
                    try:
                        season = await _tmdb_get(
                            client,
                            f"/tv/{tmdb_id}/season/{season_number}",
                            {"append_to_response": "translations"},
                        )
                        season_air_date = season.get("air_date")
                        season_poster_path = season.get("poster_path")
                        st = season.get("translations")
                        aliases.extend(_parse_translation_block(st, "season"))
                    except RuntimeError as e:
                        logger.warning("TMDB 季详情跳过: %s", e)

                poster_final = _poster_url(
                    season_poster_path or poster_path, image_base
                )
                year_final = _year_from_date(season_air_date) or year

                aliases = _merge_unique_aliases(aliases)

                credit_names = await _top_cast_names(client, "tv", tmdb_id, limit=3)

                return TmdbEnrichment(
                    found=True,
                    tmdb_id=tmdb_id,
                    tmdb_media="tv",
                    poster_url=poster_final,
                    year=year_final,
                    primary_title=primary if isinstance(primary, str) else None,
                    aliases=aliases,
                    credit_names=credit_names,
                )

            mv = await _tmdb_get(
                client,
                f"/movie/{tmdb_id}",
                {"append_to_response": "translations"},
            )
            primary = mv.get("title") or mv.get("original_title")
            poster_final = _poster_url(mv.get("poster_path"), image_base)
            year = _year_from_date(mv.get("release_date"))

            aliases.extend(_parse_translation_block(mv.get("translations"), "series"))

            alt = await _tmdb_get(client, f"/movie/{tmdb_id}/alternative_titles")
            for row in alt.get("titles") or alt.get("results") or []:
                t = row.get("title")
                region = row.get("iso_3166_1") or "XX"
                if isinstance(t, str) and t.strip():
                    aliases.append(
                        TitleAlias(
                            language=f"alt:{region}",
                            title=t.strip(),
                            scope="alternative",
                        )
                    )

            aliases = _merge_unique_aliases(aliases)

            credit_names = await _top_cast_names(client, "movie", tmdb_id, limit=3)

            return TmdbEnrichment(
                found=True,
                tmdb_id=tmdb_id,
                tmdb_media="movie",
                poster_url=poster_final,
                year=year,
                primary_title=primary if isinstance(primary, str) else None,
                aliases=aliases,
                credit_names=credit_names,
            )
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception("fetch_tmdb_enrichment 未预期异常: %s", e)
        raise RuntimeError(f"TMDB 处理失败: {e}") from e
