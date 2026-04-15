"""
搜索管线：include_domains 含 YouTube、B 站、AGE/樱花/低端/独播/欧乐等；主变体 + 三平台组合
+「剧名 在线观看 -site:youtube.com」并发；YouTube/B 站标题规则宽松，民间站须相似度≥0.8 且无成人词；
TMDB 补搜与全网约搜兜底；B 站排序加权；LLM 最多 8 条；首屏 8 条 + 可折叠更多。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_openai_base_url, get_settings
from app.schemas.interpreter import InterpretedIntent
from app.schemas.metadata import TmdbEnrichment
from app.schemas.search import PlatformTag, SearchHit, SearchResponse

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(90.0)
TAVILY_QUERY_MAX_CHARS = 400
TAVILY_MAX_RESULTS = 15
FINAL_TOP_N = 8
PRIMARY_DISPLAY_N = 8
MORE_HITS_MAX = 5
LAYER1_MIN_HITS = 3
TITLE_SIMILARITY_THRESHOLD = 0.72
SEARCH_VARIANTS_MAX = 6
MORE_HITS_NOTICE_ZH = (
    "以下链接匹配度较低或站点较陌生，请小心广告、钓鱼页与版权风险。"
)

# 标题/摘要：成人、诈骗向敏感词（命中且已匹配到剧名时整段丢弃）
TITLE_POISON_RES: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"成人|色情|无码|里番|肉番|援交|約炮|乱伦|露点|无删减完整|高潮|做爱|18禁|十八禁|R18",
        re.I,
    ),
    re.compile(
        r"\b(porn|xxx|nsfw|nudes?|naked|hentai|onlyfans|jav\d*|fc2ppv|sextape)\b",
        re.I,
    ),
)

# 无语义子串时的最低标题相似度（YouTube / B 站等宽松通道）
SEMANTIC_TITLE_SIM_MIN = 0.26

# 民间资源站：标题与参考剧名相似度须达到（含）该值，且不含成人敏感词
FOLK_SITE_TITLE_SIMILARITY_MIN = 0.8

# Tavily include_domains：YouTube、B 站 + 常用民间站（与 HIGH_TRUST_HOST_SUFFIXES 对齐）
HIGH_TRUST_INCLUDE_DOMAINS: Tuple[str, ...] = (
    "youtube.com",
    "bilibili.com",
    "b23.tv",
    "agefans.vip",
    "yhdm.tv",
    "ddys.pro",
    "duboku.tv",
    "olevod.com",
)

# 结果 URL 必须落在这些主机后缀上（含 youtu.be 短链）
HIGH_TRUST_HOST_SUFFIXES: Tuple[str, ...] = (
    "youtube.com",
    "youtu.be",
    "bilibili.com",
    "b23.tv",
    "agefans.vip",
    "yhdm.tv",
    "ddys.pro",
    "olevod.com",
    "duboku.tv",
)

# 标题或 URL 含以下片段则丢弃（短视频/社交/评分站/导览）
JUNK_TITLE_URL_MARKERS: Tuple[str, ...] = (
    "抖音",
    "douyin",
    "tiktok.com",
    "快手",
    "kuaishou",
    "微博",
    "weibo",
    "豆瓣",
    "douban",
    "justwatch",
)

# 命中则追加 YouTube 综艺专项查询
VARIETY_SHOW_KEYWORDS: Tuple[str, ...] = (
    "综艺",
    "真人秀",
    "脱口秀",
    "选秀",
    "奔跑吧",
    "奔跑",
    "歌手",
    "跨年",
    "快乐大本营",
    "天天向上",
    "极限挑战",
    "王牌对王牌",
    "非诚勿扰",
)

try:
    from tavily import TavilyClient

    _HAS_TAVILY_SDK = True
except ImportError:
    TavilyClient = None  # type: ignore[misc, assignment]
    _HAS_TAVILY_SDK = False

# —— 黑名单：百科/讨论/导览，通常无正片播放 ——
BLACKLIST_HOST_FRAGMENTS: Tuple[str, ...] = (
    "douban.com",
    "zhihu.com",
    "wikipedia.org",
    "wikimedia.org",
    "baike.baidu.com",
    "justwatch.com",
)

# —— 白名单：更可能是播放入口（域名片段，小写）——
WHITELIST_PLAYBACK_FRAGMENTS: Tuple[str, ...] = (
    "youtube.com",
    "youtu.be",
    "bilibili.com",
    "b23.tv",
    "agefans.vip",
    "agefans.",
    "yhdm.tv",
    "yhdm.",
    "ddys.pro",
    "ddys.",
    "duboku.tv",
    "duboku.",
    "olevod.com",
    "olevod.",
    "netflix.com",
)

OVERSEAS_PRIORITY_FRAGMENTS: Tuple[str, ...] = WHITELIST_PLAYBACK_FRAGMENTS + (
    "ddrk.",
)

DOMESTIC_CN_STREAMING_FRAGMENTS: Tuple[str, ...] = (
    "v.qq.com",
    "film.qq.com",
    "video.qq.com",
    "iqiyi.com",
    "iq.com",
    "youku.com",
    "mgtv.com",
    "le.com",
    "pptv.com",
    "cctv.com",
    "bilibili.com",
    "b23.tv",
    "wetv.vip",
)

_PLAYBACK_TITLE_RES: Tuple[re.Pattern[str], ...] = (
    re.compile(r"1080\s*[pP]"),
    re.compile(r"\[\s*1080\s*[pP]?\s*\]"),
    re.compile(r"高清"),
    re.compile(r"完整版"),
    re.compile(r"第\s*\d+\s*集"),
    re.compile(r"\[\s*第\s*\d+\s*集\s*\]"),
)

_CURATOR_SYSTEM = """你是视频检索结果筛选器。输入为 JSON 数组，每项含 url、title、snippet。
任务：从中挑出**最像正片播放页或聚合播放入口**的链接（最多 8 条），剔除明显仅为影评、百科、论坛、新闻、榜单、无播放的页面。
**排序优先级**：`picked_urls` 中**靠前**的应对用户意图最有把握；**哔哩哔哩（bilibili / b23.tv）**在同等相关下可优先于一般第三方站；**YouTube**在海外通常较稳。整体优先**直达播放** URL（路径含 episode、play、watch、/v/、/video/ 或路径以数字结尾）。
爱奇艺/腾讯视频：仅当 url 或标题明显为**国际版**（含 intl 等）时才优先保留；纯国内版在海外常无效，应靠后或不选。
输出要求：只返回一个 JSON 对象，不要 Markdown 代码块，不要解释。格式严格为：
{"picked_urls":["完整url1","完整url2"]}
picked_urls 中的字符串必须来自输入里的 url 字段，顺序按你认为的观看价值从高到低，最多 8 个。"""


def _normalized_host(url: str) -> str:
    try:
        h = (urlparse(url).hostname or "").lower()
        if h.startswith("www."):
            return h[4:]
        return h
    except ValueError:
        return ""


def _is_allowed_high_trust_host(url: str) -> bool:
    """仅保留配置的高信誉播放域结果。"""
    h = _normalized_host(url)
    if not h:
        return False
    if h == "youtu.be":
        return True
    return any(h == suf or h.endswith("." + suf) for suf in HIGH_TRUST_HOST_SUFFIXES if suf != "youtu.be")


def _hit_url_key(url: str) -> str:
    return url.split("?", 1)[0].rstrip("/")


def _hard_junk_url(hit: SearchHit) -> bool:
    """URL 中含明显无效域名/路径则硬过滤（与高信誉软规则无关）。"""
    low = hit.url.lower()
    for m in JUNK_TITLE_URL_MARKERS:
        if m.isascii() and m.lower() in low:
            return True
    for m in JUNK_TITLE_URL_MARKERS:
        if not m.isascii() and m in hit.url:
            return True
    return False


def _junk_in_title_snippet(hit: SearchHit) -> bool:
    """仅看标题与摘要中的干扰词（用于软过滤）。"""
    text = f"{hit.title or ''} {hit.snippet or ''}"
    low = text.lower()
    for m in JUNK_TITLE_URL_MARKERS:
        if m.isascii():
            if m.lower() in low:
                return True
        elif m in text:
            return True
    return False


def soft_filter_links(hits: List[SearchHit]) -> List[SearchHit]:
    """
    软过滤：高信誉域允许标题含干扰词保留；非高信誉且标题/摘要含干扰词则丢弃。
    URL 含硬垃圾特征的一律丢弃。
    """
    out: List[SearchHit] = []
    for h in hits:
        if _hard_junk_url(h):
            continue
        if _junk_in_title_snippet(h) and not _is_allowed_high_trust_host(h.url):
            continue
        out.append(h)
    return out


def filter_links(hits: List[SearchHit]) -> List[SearchHit]:
    """兼容旧名，等同 soft_filter_links。"""
    return soft_filter_links(hits)


def _is_junk_hit(hit: SearchHit) -> bool:
    """硬+软合并判断（单测/调试）。"""
    if _hard_junk_url(hit):
        return True
    return _junk_in_title_snippet(hit) and not _is_allowed_high_trust_host(hit.url)


def _normalize_similarity_blob(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip().lower())


def _similarity_reference_intent(intent: InterpretedIntent, meta: TmdbEnrichment) -> str:
    parts: List[str] = []
    if meta.found and meta.primary_title and str(meta.primary_title).strip():
        parts.append(str(meta.primary_title).strip())
    t = (intent.title or "").strip()
    if t and t not in parts:
        parts.append(t)
    ot = (intent.original_title or "").strip()
    if ot and ot not in parts:
        parts.append(ot)
    return " ".join(parts)


def _reference_similarity_forms(reference: str) -> List[str]:
    """同一剧名的多种归一化形式（如 01 ↔ 零一），用于模糊匹配。"""
    raw = (reference or "").strip()
    if not raw:
        return []
    forms: List[str] = []
    seen: Set[str] = set()

    def push(s: str) -> None:
        n = _normalize_similarity_blob(s)
        if len(n) >= 2 and n not in seen:
            seen.add(n)
            forms.append(n)

    push(raw)
    if re.search(r"01|０１", raw):
        push(re.sub(r"01|０１", "零一", raw, count=1))
    if "零一" in raw:
        push(raw.replace("零一", "01", 1))
    return forms


def title_similarity_score(reference: str, hit: SearchHit) -> float:
    """标题/摘要与参考剧名的相似度，用于第二层全网约搜过滤（>=0.8 保留）。"""
    alts = _reference_similarity_forms(reference)
    if not alts:
        return 0.0
    title = _normalize_similarity_blob(hit.title or "")
    blob = _normalize_similarity_blob(f"{hit.title or ''}{hit.snippet or ''}")
    best = 0.0
    for ref in alts:
        if ref in blob:
            best = max(best, 0.9)
        if title:
            best = max(best, SequenceMatcher(None, ref, title).ratio())
        if blob:
            best = max(best, SequenceMatcher(None, ref, blob).ratio())
    return best


def _text_has_poison(text: str) -> bool:
    if not (text or "").strip():
        return False
    for rx in TITLE_POISON_RES:
        if rx.search(text):
            return True
    return False


def _is_tier1_trust_host(url: str) -> bool:
    """一级信任：重排加分（YouTube / 低端 / 独播库）。"""
    h = _normalized_host(url)
    if not h:
        return False
    if h == "youtu.be":
        return True
    if h == "youtube.com" or h.endswith(".youtube.com"):
        return True
    if h == "ddys.pro" or h.endswith(".ddys.pro"):
        return True
    if h == "duboku.tv" or h.endswith(".duboku.tv"):
        return True
    return False


def _is_youtube_host_url(url: str) -> bool:
    h = _normalized_host(url)
    if h == "youtu.be":
        return True
    return h == "youtube.com" or h.endswith(".youtube.com")


def _core_display_names(intent: InterpretedIntent, meta: TmdbEnrichment) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for s in (
        meta.primary_title if meta.found else None,
        intent.title,
        intent.original_title,
    ):
        if s and isinstance(s, str):
            t = s.strip()
            if len(t) >= 2 and t not in seen:
                seen.add(t)
                out.append(t)
    for a in (meta.aliases or [])[:16]:
        if a.title and a.title.strip():
            t = a.title.strip()
            if len(t) >= 2 and t not in seen:
                seen.add(t)
                out.append(t)
    return sorted(out, key=len, reverse=True)


def _title_has_core_substring(title: Optional[str], core_names: Sequence[str]) -> bool:
    if not title or not core_names:
        return False
    nt = _normalize_similarity_blob(title)
    low = title.lower()
    for n in core_names:
        if len(n) < 2:
            continue
        if _normalize_similarity_blob(n) in nt:
            return True
        if n.lower() in low:
            return True
    return False


def _text_has_any_core_name(text: Optional[str], core_names: Sequence[str]) -> bool:
    """标题或摘要中是否出现任一剧名子串（不要求整标题一致）。"""
    return _title_has_core_substring(text or "", core_names)


def _hit_passes_semantic_title(
    hit: SearchHit,
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
) -> bool:
    """标题或摘要含剧名且无成人黑名单即通过；否则用较低相似度阈值兜底。"""
    names = _core_display_names(intent, meta)
    ref = _similarity_reference_intent(intent, meta)
    if not names:
        return True
    blob_all = f"{hit.title or ''} {hit.snippet or ''}"
    core_ok = _text_has_any_core_name(hit.title, names) or _text_has_any_core_name(
        blob_all, names,
    )
    if core_ok:
        return not _text_has_poison(blob_all)
    if _text_has_poison(hit.title or ""):
        return False
    return title_similarity_score(ref, hit) >= SEMANTIC_TITLE_SIM_MIN


def _hit_passes_semantic_by_host(
    hit: SearchHit,
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
) -> bool:
    """
    标题 + 域名双重过滤：YouTube / B 站在无毒前提下放宽标题匹配（检索与白名单已约束）；
    民间站（AGE、樱花、低端、独播、欧乐等）须标题相似度 ≥ 阈值且无毒。
    """
    ref = _similarity_reference_intent(intent, meta)
    blob_all = f"{hit.title or ''} {hit.snippet or ''}"
    if _text_has_poison(blob_all):
        return False
    if _is_youtube_host_url(hit.url) or _is_bilibili_host_url(hit.url):
        return True
    if _is_allowed_high_trust_host(hit.url):
        return title_similarity_score(ref, hit) >= FOLK_SITE_TITLE_SIMILARITY_MIN
    return False


def _filter_hits_semantic_and_tier(
    hits: List[SearchHit],
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
) -> List[SearchHit]:
    return [h for h in hits if _hit_passes_semantic_by_host(h, intent, meta)]


def _is_bilibili_host_url(url: str) -> bool:
    h = _normalized_host(url)
    return "bilibili.com" in h or h == "b23.tv" or h.endswith(".bilibili.com")


def _enrich_hit_source_tags(hit: SearchHit) -> SearchHit:
    host = _normalized_host(hit.url)
    display = host
    if not display:
        try:
            display = (urlparse(hit.url).hostname or "").lower()
        except ValueError:
            display = ""
    if display.startswith("www."):
        display = display[4:]
    yt = _is_youtube_host_url(hit.url)
    bili = _is_bilibili_host_url(hit.url)
    official = yt or bili
    return hit.model_copy(
        update={
            "display_domain": display or None,
            "official_recommend_badge": official,
            "youtube_recommend_badge": yt and not bili,
            "bilibili_hot_badge": bili,
            "third_party_ad_caution": not official,
        }
    )


def _apply_source_enrichment(hits: Sequence[SearchHit]) -> List[SearchHit]:
    return [_enrich_hit_source_tags(h) for h in hits]


def build_precision_refined_queries(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
    *,
    overseas_mode: bool = False,
) -> List[str]:
    """第二轮并发：剧名 + TMDB 年份 / 主演，收窄泛匹配。"""
    if not meta.found:
        return []
    if meta.primary_title and str(meta.primary_title).strip():
        primary = str(meta.primary_title).strip()
    else:
        primary = (intent.title or "").strip()
    if not primary:
        return []

    seen: Set[str] = set()
    out: List[str] = []

    def add(raw: str) -> None:
        q = truncate_search_query(raw.strip())
        if q and q not in seen:
            seen.add(q)
            out.append(q)

    y = meta.year
    if y is not None:
        add(f"{primary} {y} 在线播放")
        add(f"{primary} {y} 观看 完整版")
    for name in (meta.credit_names or [])[:2]:
        ns = str(name).strip()
        if ns:
            add(f"{primary} {ns} 在线观看")
    ot = (intent.original_title or "").strip()
    if ot and ot != primary and len(ot) <= 60 and y is not None:
        add(f"{ot} {y} 在线播放")

    if overseas_mode:
        out = [truncate_search_query(q + " 海外观看") for q in out]

    return out[:4]


def build_three_platform_focus_queries(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
) -> List[str]:
    """与主变体并行：剧名 + B站 / 在线观看 / AGE动漫，拓宽二次元站召回。"""
    core = _primary_show_title(intent, meta) or (intent.title or "").strip()
    if len(core) < 1:
        return []
    return [
        truncate_search_query(f"{core} B站"),
        truncate_search_query(f"{core} 在线观看"),
        truncate_search_query(f"{core} AGE动漫"),
    ]


async def _tavily_three_platform_queries(queries: Sequence[str]) -> List[SearchHit]:
    if not queries:
        return []
    mr = min(10, TAVILY_MAX_RESULTS)
    tasks = [
        _search_tavily(
            q,
            include_domains=list(HIGH_TRUST_INCLUDE_DOMAINS),
            max_results=mr,
        )
        for q in queries
    ]
    merged: List[SearchHit] = []
    for p in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(p, Exception):
            logger.warning("Tavily 三组合查询失败: %s", p)
            continue
        merged.extend(p)
    return merged


async def _serper_three_platform_queries(queries: Sequence[str]) -> List[SearchHit]:
    if not queries:
        return []
    tasks = [_search_serper(q, restrict_sites=True) for q in queries]
    merged: List[SearchHit] = []
    for p in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(p, Exception):
            logger.warning("Serper 三组合查询失败: %s", p)
            continue
        merged.extend(p)
    return merged


def build_folk_minus_youtube_query(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
) -> Optional[str]:
    """民间站导向：排除 YouTube 的检索词；实际域范围仍由 include_domains 约束。"""
    core = _primary_show_title(intent, meta) or (intent.title or "").strip()
    if not core:
        return None
    return truncate_search_query(f"{core} 在线观看 -site:youtube.com")


async def _tavily_folk_minus_youtube_queries(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
) -> List[SearchHit]:
    q = build_folk_minus_youtube_query(intent, meta)
    if not q:
        return []
    try:
        return await _search_tavily(
            q,
            include_domains=list(HIGH_TRUST_INCLUDE_DOMAINS),
            max_results=min(10, TAVILY_MAX_RESULTS),
        )
    except Exception as e:
        logger.warning("Tavily 民间向（-site:youtube.com）查询失败: %s", e)
        return []


async def _serper_folk_minus_youtube_queries(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
) -> List[SearchHit]:
    q = build_folk_minus_youtube_query(intent, meta)
    if not q:
        return []
    try:
        return await _search_serper(q, restrict_sites=True)
    except Exception as e:
        logger.warning("Serper 民间向（-site:youtube.com）查询失败: %s", e)
        return []


def _merged_to_trust_pool(
    merged: List[SearchHit],
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
) -> List[SearchHit]:
    h = [x for x in merged if _is_allowed_high_trust_host(x.url)]
    h = _filter_blacklist(h)
    return _filter_hits_semantic_and_tier(h, intent, meta)


def _direct_url_rank_bonus(url: str) -> float:
    """直达播放路径加权：episode/play/watch 或路径以数字结尾。"""
    try:
        path = (urlparse(url).path or "").lower()
    except ValueError:
        return 0.0
    bonus = 0.0
    if re.search(r"/(?:episode|play|watch|v|video)(?:/|$|[?#])", path):
        bonus += 28.0
    if re.search(r"/\d+/?$", path):
        bonus += 18.0
    return bonus


def _primary_show_title(intent: InterpretedIntent, meta: TmdbEnrichment) -> str:
    if meta.found and meta.primary_title and str(meta.primary_title).strip():
        return str(meta.primary_title).strip()
    return (intent.title or "").strip()


def _should_youtube_variety_boost(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
) -> bool:
    """综艺类：追加 site:youtube.com 完整版 专项检索。"""
    blob = " ".join(
        [
            _primary_show_title(intent, meta),
            (intent.original_title or "").strip(),
            (intent.season_official_name or "").strip(),
        ]
    )
    if not blob.strip():
        return False
    return any(k in blob for k in VARIETY_SHOW_KEYWORDS)


def build_youtube_variety_query(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
) -> str:
    title = _primary_show_title(intent, meta)
    if not title:
        title = (intent.title or "").strip() or (intent.original_title or "").strip()
    inner = title.replace('"', " ").strip()
    return truncate_search_query(f'site:youtube.com "{inner}" 完整版')


def _serper_site_restriction_suffix() -> str:
    sites = " OR ".join(f"site:{d}" for d in HIGH_TRUST_INCLUDE_DOMAINS)
    return f" ({sites})"


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def _is_blacklisted_url(url: str) -> bool:
    h = _normalized_host(url)
    return any(b in h for b in BLACKLIST_HOST_FRAGMENTS)


def _is_whitelist_playback(url: str) -> bool:
    h = _normalized_host(url)
    return any(w in h for w in WHITELIST_PLAYBACK_FRAGMENTS)


def _is_overseas_priority_url(url: str) -> bool:
    h = _normalized_host(url)
    return any(f in h for f in OVERSEAS_PRIORITY_FRAGMENTS)


def _is_domestic_cn_streaming_url(url: str) -> bool:
    h = _normalized_host(url)
    if not h:
        return False
    if h == "qq.com" or h.endswith(".qq.com"):
        return True
    return any(frag in h for frag in DOMESTIC_CN_STREAMING_FRAGMENTS)


def _is_iqiyi_or_tencent(url: str) -> bool:
    u = url.lower()
    h = _normalized_host(url)
    if "iqiyi" in h or "iq.com" in h:
        return True
    if "qq.com" in h or "v.qq" in h:
        return True
    return "iqiyi" in u or "v.qq" in u


def _has_intl_marker(url: str, title: Optional[str]) -> bool:
    blob = f"{url} {title or ''}".lower()
    return "intl" in blob


def truncate_search_query(q: str, max_len: int = TAVILY_QUERY_MAX_CHARS) -> str:
    q = " ".join(q.split())
    if len(q) <= max_len:
        return q
    out = q[:max_len].rstrip()
    logger.warning("搜索词过长已截断: 原长=%s -> %s 字符", len(q), max_len)
    return out


def build_search_query(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
    *,
    overseas_mode: bool = False,
) -> str:
    """剧名 + 多组观看相关关键词；中文资源追加常用站名后缀。"""
    parts: List[str] = []

    if meta.found and meta.primary_title and str(meta.primary_title).strip():
        parts.append(str(meta.primary_title).strip())
    else:
        t = (intent.title or "").strip()
        if t:
            parts.append(t)

    ot = (intent.original_title or "").strip()
    if ot and ot not in parts and len(ot) <= 60:
        parts.append(ot)

    if intent.season is not None:
        parts.append(f"第{intent.season}季")
    if intent.episode is not None:
        parts.append(f"第{intent.episode}集")
    arc = (intent.season_official_name or "").strip()
    if arc and len(arc) <= 40:
        parts.append(arc)

    parts.extend(["在线观看", "完整版", "多线播放"])

    blob = " ".join(parts)
    if _has_cjk(blob) or (intent.title and _has_cjk(intent.title)):
        parts.extend(["低端影视", "独播库", "泥巴影院", "欧乐影院"])

    parts.append("正版")

    if overseas_mode:
        parts.extend(
            [
                "海外可用",
                "无海外限制",
                "YouTube",
                "Gimy",
                "Duboku",
            ]
        )

    return " ".join(parts)


def build_search_query_variants(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
    *,
    overseas_mode: bool = False,
) -> List[str]:
    """
    多组检索词：主查询 + 轻量「剧名+在线」+ 原文名 + 常见数字/别名字面变体，
    提高如「假面骑士01」类条目的召回。
    """
    seen: Set[str] = set()
    out: List[str] = []

    def add(raw: str) -> None:
        q = truncate_search_query(raw.strip())
        if not q or q in seen:
            return
        seen.add(q)
        out.append(q)

    add(build_search_query(intent, meta, overseas_mode=overseas_mode))

    core = _primary_show_title(intent, meta) or (intent.title or "").strip()
    ot = (intent.original_title or "").strip()

    if core:
        add(f"{core} 在线观看")
        add(f"{core} 完整版 在线")
        if "假面骑士" in core and re.search(r"01|０１", core):
            alt = re.sub(r"01|０１", "零一", core, count=1)
            if alt != core:
                add(f"{alt} 在线观看")
            add("Kamen Rider Zero-One 在线观看 完整版")
        m_nums = re.findall(r"\d{2,}", core)
        if m_nums and "假面骑士" not in core:
            spaced = re.sub(r"(\d+)", r" \1 ", core)
            spaced = " ".join(spaced.split())
            if spaced != core:
                add(f"{spaced} 在线观看")

    if ot and ot != core and len(ot) <= 80:
        add(f"{ot} online watch 完整版")

    return out[:SEARCH_VARIANTS_MAX]


def classify_platform(url: str) -> PlatformTag:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return "third_party"
    if not host:
        return "third_party"
    if host.startswith("www."):
        host = host[4:]
    if "bilibili.com" in host or host == "b23.tv":
        return "bilibili"
    if "netflix.com" in host:
        return "netflix"
    if "youtube.com" in host or host == "youtu.be":
        return "youtube"
    mainstream_markers = (
        "crunchyroll.com",
        "primevideo.com",
        "amazon.com",
        "disneyplus.com",
        "hulu.com",
        "max.com",
        "iq.com",
        "iqiyi.com",
        "v.qq.com",
        "youku.com",
        "mgtv.com",
        "wetv.vip",
        "apple.com",
        "tv.apple.com",
    )
    if any(m in host for m in mainstream_markers):
        return "mainstream_other"
    return "third_party"


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _message_text(msg: BaseMessage) -> str:
    if isinstance(msg, AIMessage) or hasattr(msg, "content"):
        c: Any = getattr(msg, "content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            chunks: List[str] = []
            for block in c:
                if isinstance(block, str):
                    chunks.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    chunks.append(str(block.get("text", "")))
            return "".join(chunks)
    return str(msg)


def _title_snippet_boost(hit: SearchHit, intent: InterpretedIntent) -> float:
    text = f"{hit.title or ''} {hit.snippet or ''}"
    s = 0.0
    for rx in _PLAYBACK_TITLE_RES:
        if rx.search(text):
            s += 22.0
    if intent.episode is not None and re.search(
        rf"第\s*{intent.episode}\s*集", text
    ):
        s += 18.0
    if re.search(r"播放|正片|全集|观看", text):
        s += 8.0
    return s


def _playback_rank_score(hit: SearchHit, intent: InterpretedIntent) -> float:
    """分数越高越优先（智能重排）。"""
    url = hit.url
    score = 0.0
    if _is_whitelist_playback(url):
        score += 120.0
    if _is_youtube_host_url(url):
        score += 35.0
    if _is_bilibili_host_url(url):
        score += 88.0
    if _is_tier1_trust_host(url):
        score += 22.0
    score += _title_snippet_boost(hit, intent)

    if _is_iqiyi_or_tencent(url):
        if _has_intl_marker(url, hit.title):
            score += 45.0
        else:
            score -= 75.0

    if _is_domestic_cn_streaming_url(url) and not _has_intl_marker(url, hit.title):
        score -= 25.0

    score += _direct_url_rank_bonus(url)

    return score


def _smart_rerank(hits: List[SearchHit], intent: InterpretedIntent) -> List[SearchHit]:
    indexed = list(enumerate(hits))
    indexed.sort(
        key=lambda it: (-_playback_rank_score(it[1], intent), it[0]),
    )
    return [h for _, h in indexed]


def _hits_from_tavily_payload(data: Dict[str, Any]) -> List[SearchHit]:
    out: List[SearchHit] = []
    for row in data.get("results") or []:
        if not isinstance(row, dict):
            continue
        u = row.get("url")
        if not u or not isinstance(u, str):
            continue
        out.append(
            SearchHit(
                url=u,
                title=row.get("title") if isinstance(row.get("title"), str) else None,
                snippet=(
                    row.get("content")
                    if isinstance(row.get("content"), str)
                    else (
                        row.get("snippet") if isinstance(row.get("snippet"), str) else None
                    )
                ),
                platform=classify_platform(u),
                source="tavily",
            )
        )
    return out


def _tavily_exclude_domains() -> List[str]:
    return [
        "douban.com",
        "zhihu.com",
        "wikipedia.org",
        "wikimedia.org",
        "baike.baidu.com",
        "justwatch.com",
    ]


async def _search_tavily_http(
    api_key: str,
    query: str,
    *,
    include_domains: Optional[Sequence[str]] = None,
    max_results: Optional[int] = None,
) -> Dict[str, Any]:
    mr = int(max_results) if max_results is not None else TAVILY_MAX_RESULTS
    inc = (
        list(include_domains)
        if include_domains is not None
        else list(HIGH_TRUST_INCLUDE_DOMAINS)
    )
    payload: Dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "max_results": mr,
    }
    if inc:
        payload["include_domains"] = inc
    else:
        payload["exclude_domains"] = _tavily_exclude_domains()
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        r = await client.post("https://api.tavily.com/search", json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "")[:1200]
            logger.exception(
                "Tavily HTTP %s body[:800]=%r",
                e.response.status_code,
                body[:800],
            )
            raise RuntimeError(
                f"Tavily HTTP {e.response.status_code}：{body[:400]}"
            ) from e
        except httpx.RequestError as e:
            logger.exception("Tavily 网络错误: %s", e)
            raise RuntimeError(f"Tavily 网络错误: {e}") from e
        try:
            data = r.json()
        except ValueError as e:
            logger.exception("Tavily 响应非 JSON: %r", (r.text or "")[:500])
            raise RuntimeError("Tavily 返回非 JSON") from e
    if not isinstance(data, dict):
        raise RuntimeError("Tavily 返回格式异常（非对象）")
    return data


async def _search_tavily(
    query: str,
    *,
    include_domains: Optional[Sequence[str]] = None,
    max_results: Optional[int] = None,
) -> List[SearchHit]:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise RuntimeError("未设置 TAVILY_API_KEY，无法使用 Tavily 搜索。")
    mr = int(max_results) if max_results is not None else TAVILY_MAX_RESULTS
    inc = (
        list(include_domains)
        if include_domains is not None
        else list(HIGH_TRUST_INCLUDE_DOMAINS)
    )
    try:
        if _HAS_TAVILY_SDK and TavilyClient is not None:

            def _run_sdk() -> Dict[str, Any]:
                client = TavilyClient(api_key=settings.tavily_api_key)
                kwargs: Dict[str, Any] = {
                    "search_depth": "advanced",
                    "include_answer": False,
                    "max_results": mr,
                }
                if inc:
                    kwargs["include_domains"] = inc
                else:
                    kwargs["exclude_domains"] = _tavily_exclude_domains()
                out = client.search(query, **kwargs)
                return out if isinstance(out, dict) else {"results": []}

            data = await asyncio.to_thread(_run_sdk)
        else:
            data = await _search_tavily_http(
                settings.tavily_api_key,
                query,
                include_domains=include_domains,
                max_results=max_results,
            )
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception("Tavily 调用失败 query=%r: %s", query[:300], e)
        raise RuntimeError(f"Tavily 搜索失败: {e}") from e

    err = data.get("error") or data.get("detail")
    if err:
        logger.error("Tavily 返回错误字段: %s", data)
        raise RuntimeError(f"Tavily 错误: {err}")
    return _hits_from_tavily_payload(data)


async def _search_serper(query: str, *, restrict_sites: bool = True) -> List[SearchHit]:
    settings = get_settings()
    if not settings.serper_api_key:
        raise RuntimeError("未设置 SERPER_API_KEY，无法使用 Serper 搜索。")
    suffix = _serper_site_restriction_suffix() if restrict_sites else ""
    q = truncate_search_query((query + suffix).strip())
    headers = {"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"q": q, "num": min(TAVILY_MAX_RESULTS, 20)}
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            try:
                data = r.json()
            except ValueError as e:
                logger.exception("Serper 响应非 JSON: %r", (r.text or "")[:500])
                raise RuntimeError("Serper 返回非 JSON") from e
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:800]
        logger.exception("Serper HTTP %s: %s", e.response.status_code, body)
        raise RuntimeError(f"Serper HTTP {e.response.status_code}: {body[:300]}") from e
    except httpx.RequestError as e:
        logger.exception("Serper 网络错误: %s", e)
        raise RuntimeError(f"Serper 网络错误: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError("Serper 返回格式异常")
    out: List[SearchHit] = []
    for row in data.get("organic") or []:
        if not isinstance(row, dict):
            continue
        u = row.get("link")
        if not u or not isinstance(u, str):
            continue
        out.append(
            SearchHit(
                url=u,
                title=row.get("title") if isinstance(row.get("title"), str) else None,
                snippet=row.get("snippet") if isinstance(row.get("snippet"), str) else None,
                platform=classify_platform(u),
                source="serper",
            )
        )
    return out


def _filter_blacklist(hits: List[SearchHit]) -> List[SearchHit]:
    return [h for h in hits if not _is_blacklisted_url(h.url)]


def _apply_overseas_annotations_only(
    hits: List[SearchHit],
    overseas_mode: bool,
) -> Tuple[List[SearchHit], bool]:
    """不改变顺序，仅标注国内站与 VPN 提示阈值。"""
    if not overseas_mode or not hits:
        return hits, False
    vpn_note = "可能需要回国 VPN"
    out: List[SearchHit] = []
    domestic_flags: List[bool] = []
    for h in hits:
        if _is_domestic_cn_streaming_url(h.url):
            domestic_flags.append(True)
            out.append(h.model_copy(update={"access_note": vpn_note}))
        else:
            domestic_flags.append(False)
            out.append(h)
    n = len(out)
    domestic_n = sum(domestic_flags)
    show_hint = n >= 3 and domestic_n / n >= 0.5
    return out, show_hint


async def _llm_pick_top_playback(
    candidates: Sequence[SearchHit],
    intent: InterpretedIntent,
) -> Optional[List[SearchHit]]:
    """LLM 从候选中最多选 5 条；失败返回 None。"""
    settings = get_settings()
    if not settings.search_llm_curate:
        return None
    if not settings.openai_api_key:
        return None
    cand = list(candidates)[: min(24, len(candidates))]
    if not cand:
        return None
    rows = [
        {
            "url": h.url,
            "title": h.title,
            "snippet": (h.snippet or "")[:280],
        }
        for h in cand
    ]
    human = (
        f"用户想找：{intent.title!s} "
        f"季={intent.season!s} 集={intent.episode!s}\n\n"
        f"候选 JSON：\n{json.dumps(rows, ensure_ascii=False)}"
    )
    try:
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=get_openai_base_url(),
            temperature=0.1,
        )
        raw_msg = await llm.ainvoke(
            [
                SystemMessage(content=_CURATOR_SYSTEM),
                HumanMessage(content=human),
            ]
        )
        text = _message_text(raw_msg)
        data = json.loads(_strip_json_fences(text))
    except Exception as e:
        logger.warning("搜索精选 LLM 解析失败: %s", e)
        return None

    urls_raw = data.get("picked_urls")
    if not isinstance(urls_raw, list):
        return None
    picked: List[str] = []
    for u in urls_raw:
        if isinstance(u, str) and u.strip():
            picked.append(u.strip())
        if len(picked) >= FINAL_TOP_N:
            break
    url_to_hit = {h.url: h for h in cand}
    # 归一化：允许模型少写协议
    result: List[SearchHit] = []
    for u in picked:
        if u in url_to_hit:
            result.append(url_to_hit[u])
            continue
        for key, hit in url_to_hit.items():
            if u in key or key.endswith(u) or u.endswith(key.split("://", 1)[-1]):
                result.append(hit)
                break
    if not result:
        return None
    seen: Set[str] = set()
    deduped: List[SearchHit] = []
    for h in result:
        k = h.url.split("?", 1)[0].rstrip("/")
        if k in seen:
            continue
        seen.add(k)
        deduped.append(h)
    return deduped[:FINAL_TOP_N]


def _fallback_top_n(hits: List[SearchHit], n: int) -> List[SearchHit]:
    return hits[:n]


def _dedupe_hits(hits: List[SearchHit]) -> List[SearchHit]:
    seen: Set[str] = set()
    out: List[SearchHit] = []
    for hit in hits:
        key = _hit_url_key(hit.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


async def _tavily_layer1_variants(variants: Sequence[str]) -> List[SearchHit]:
    mr = min(8, TAVILY_MAX_RESULTS)
    tasks = [
        _search_tavily(
            q,
            include_domains=list(HIGH_TRUST_INCLUDE_DOMAINS),
            max_results=mr,
        )
        for q in variants
    ]
    merged: List[SearchHit] = []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for p in results:
        if isinstance(p, Exception):
            logger.warning("Tavily 白名单变体查询失败: %s", p)
            continue
        merged.extend(p)
    return merged


async def _serper_layer1_variants(variants: Sequence[str]) -> List[SearchHit]:
    tasks = [_search_serper(q, restrict_sites=True) for q in variants]
    merged: List[SearchHit] = []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for p in results:
        if isinstance(p, Exception):
            logger.warning("Serper 白名单变体查询失败: %s", p)
            continue
        merged.extend(p)
    return merged


async def _layer2_extended_open_tavily(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
    *,
    overseas_mode: bool,
    existing_keys: Set[str],
) -> List[SearchHit]:
    ref = _similarity_reference_intent(intent, meta)
    if not ref.strip():
        return []
    qs = build_search_query_variants(intent, meta, overseas_mode=overseas_mode)[:2]
    merged: List[SearchHit] = []
    for q in qs:
        try:
            part = await _search_tavily(
                q,
                include_domains=[],
                max_results=TAVILY_MAX_RESULTS,
            )
        except Exception as e:
            logger.warning("第二层 Tavily 全网约搜失败: %s", e)
            continue
        for h in part:
            if _hit_url_key(h.url) in existing_keys:
                continue
            if title_similarity_score(ref, h) < TITLE_SIMILARITY_THRESHOLD:
                continue
            merged.append(h)
    merged = _dedupe_hits(merged)
    merged = soft_filter_links(merged)
    return [h for h in merged if not _is_blacklisted_url(h.url)]


async def _layer2_extended_open_serper(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
    *,
    overseas_mode: bool,
    existing_keys: Set[str],
) -> List[SearchHit]:
    ref = _similarity_reference_intent(intent, meta)
    if not ref.strip():
        return []
    qs = build_search_query_variants(intent, meta, overseas_mode=overseas_mode)[:2]
    merged: List[SearchHit] = []
    for q in qs:
        try:
            part = await _search_serper(q, restrict_sites=False)
        except Exception as e:
            logger.warning("第二层 Serper 全网约搜失败: %s", e)
            continue
        for h in part:
            if _hit_url_key(h.url) in existing_keys:
                continue
            if title_similarity_score(ref, h) < TITLE_SIMILARITY_THRESHOLD:
                continue
            merged.append(h)
    merged = _dedupe_hits(merged)
    merged = soft_filter_links(merged)
    return [h for h in merged if not _is_blacklisted_url(h.url)]


async def search_watch_candidates(
    intent: InterpretedIntent,
    meta: TmdbEnrichment,
    *,
    overseas_mode: bool = False,
) -> SearchResponse:
    try:
        settings = get_settings()
        variants = build_search_query_variants(
            intent, meta, overseas_mode=overseas_mode
        )
        if not variants or not str(variants[0]).strip():
            return SearchResponse(
                hits=[],
                more_hits=[],
                more_hits_notice=None,
                show_domestic_vpn_hint=False,
            )

        provider = (settings.search_provider or "tavily").lower().strip()
        raw: List[SearchHit] = []
        tri_q = build_three_platform_focus_queries(intent, meta)
        if provider == "serper":
            v_task = asyncio.create_task(_serper_layer1_variants(variants))
            t_task = (
                asyncio.create_task(_serper_three_platform_queries(tri_q))
                if tri_q
                else None
            )
            f_task = asyncio.create_task(_serper_folk_minus_youtube_queries(intent, meta))
            raw_v = await v_task
            raw_t = await t_task if t_task else []
            raw_f = await f_task
            raw = _dedupe_hits(raw_f + raw_t + raw_v)
        else:
            v_task = asyncio.create_task(_tavily_layer1_variants(variants))
            t_task = (
                asyncio.create_task(_tavily_three_platform_queries(tri_q))
                if tri_q
                else None
            )
            f_task = asyncio.create_task(_tavily_folk_minus_youtube_queries(intent, meta))
            raw_v = await v_task
            raw_t = await t_task if t_task else []
            raw_f = await f_task
            raw = _dedupe_hits(raw_f + raw_t + raw_v)
            if _should_youtube_variety_boost(intent, meta):
                yq = build_youtube_variety_query(intent, meta)
                if yq.strip():
                    try:
                        raw_yt = await _search_tavily(
                            yq,
                            include_domains=["youtube.com"],
                            max_results=min(10, TAVILY_MAX_RESULTS),
                        )
                        raw = raw_yt + raw
                    except Exception as e:
                        logger.warning("YouTube 综艺补充搜索失败，沿用主结果: %s", e)

        merged = _dedupe_hits(raw)
        merged = soft_filter_links(merged)
        trust = _merged_to_trust_pool(merged, intent, meta)

        if len(trust) < LAYER1_MIN_HITS and meta.found:
            pq = build_precision_refined_queries(
                intent, meta, overseas_mode=overseas_mode
            )
            if pq:
                if provider == "serper":
                    extra_p = await _serper_layer1_variants(pq)
                else:
                    extra_p = await _tavily_layer1_variants(pq)
                merged = _dedupe_hits(merged + extra_p)
                merged = soft_filter_links(merged)
                trust = _merged_to_trust_pool(merged, intent, meta)

        if len(trust) < LAYER1_MIN_HITS:
            keys = {_hit_url_key(h.url) for h in trust}
            if provider == "serper":
                extra = await _layer2_extended_open_serper(
                    intent,
                    meta,
                    overseas_mode=overseas_mode,
                    existing_keys=keys,
                )
            else:
                extra = await _layer2_extended_open_tavily(
                    intent,
                    meta,
                    overseas_mode=overseas_mode,
                    existing_keys=keys,
                )
            merged = _dedupe_hits(merged + extra)
            merged = soft_filter_links(merged)
            trust = _merged_to_trust_pool(merged, intent, meta)

        merged_pre = trust

        if not merged_pre:
            return SearchResponse(
                hits=[],
                more_hits=[],
                more_hits_notice=None,
                show_domestic_vpn_hint=False,
            )

        ranked = _smart_rerank(merged_pre, intent)
        curated = await _llm_pick_top_playback(ranked, intent)

        cap = PRIMARY_DISPLAY_N + MORE_HITS_MAX
        ordered: List[SearchHit]
        pick_keys: Set[str]
        if curated:
            ordered = list(curated)
            pick_keys = {_hit_url_key(h.url) for h in curated}
            for h in ranked:
                if len(ordered) >= cap:
                    break
                k = _hit_url_key(h.url)
                if k in pick_keys:
                    continue
                pick_keys.add(k)
                ordered.append(h)
        else:
            ordered = list(ranked[:cap])

        ordered_annotated, show_hint = _apply_overseas_annotations_only(
            ordered, overseas_mode
        )
        ordered_annotated = _apply_source_enrichment(ordered_annotated)
        hits_out = ordered_annotated[:PRIMARY_DISPLAY_N]
        more_out = ordered_annotated[PRIMARY_DISPLAY_N:cap]

        return SearchResponse(
            hits=hits_out,
            more_hits=more_out,
            more_hits_notice=MORE_HITS_NOTICE_ZH if more_out else None,
            show_domestic_vpn_hint=show_hint,
        )
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception("search_watch_candidates 未预期异常: %s", e)
        raise RuntimeError(f"搜索阶段失败: {e}") from e
