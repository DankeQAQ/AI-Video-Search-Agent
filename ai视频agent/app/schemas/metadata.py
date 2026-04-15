from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TitleAlias(BaseModel):
    """TMDB 返回的某一语言下的标题别名。"""

    language: str = Field(..., description="ISO 639-1 或 TMDB 返回的语言/地区标识")
    title: str
    scope: Literal["series", "season", "alternative"] = Field(
        ...,
        description="series=整剧译名；season=该季译名；alternative=备选标题",
    )


class TmdbEnrichment(BaseModel):
    """与 TMDB 对齐后的元数据，供搜索与展示使用。"""

    found: bool = Field(..., description="是否在 TMDB 命中条目")
    tmdb_id: Optional[int] = None
    tmdb_media: Optional[Literal["tv", "movie"]] = None
    poster_url: Optional[str] = Field(None, description="正式海报（优先季海报，其次整剧）")
    year: Optional[int] = Field(None, description="首播年：有季时优先取该季首播年")
    primary_title: Optional[str] = Field(None, description="TMDB 默认语言下的主标题")
    aliases: List[TitleAlias] = Field(default_factory=list, description="多语言别名与备选标题")
    credit_names: List[str] = Field(
        default_factory=list,
        description="TMDB credits 中靠前的主演姓名，用于精细化搜索词。",
    )
